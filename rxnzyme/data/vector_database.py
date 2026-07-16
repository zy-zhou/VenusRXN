import atexit
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

index_params = {
    'index_type': 'HNSW',
    'metric_type': 'IP',
    'params': {'M': 16, 'efConstruction': 200},
}

search_params = {
    'metric_type': 'IP',
    'params': {'ef': 128},
}

grpc_options = {
    'grpc.keepalive_permit_without_calls': False,
    'grpc.keepalive_time_ms': 100000,
    'grpc.keepalive_timeout_ms': 20000,
}

MILVUS_LITE_ADDR = '127.0.0.1:19530'

def is_local_uri(uri):
    return '://' not in str(uri)

def start_milvus_lite_server(db_path, addr=MILVUS_LITE_ADDR):
    '''
    Start a local milvus-lite gRPC server (single-process owner of the DB file),
    and return a pymilvus-compatible URI for clients.
    '''
    try:
        from milvus_lite.server import Server
    except Exception as e:
        raise RuntimeError(
            'milvus-lite local DB path requires milvus-lite server mode for concurrency. '
            'Please install milvus-lite or pass a remote Milvus URI instead.'
        ) from e

    server = Server(db_path, addr)
    server.start()
    atexit.register(server.stop)
    return f'http://{addr}'

class MilvusStorage:
    def __init__(
        self,
        uri,
        token='',
        db_name='',
        collection='embeddings',
        vector_dim=256,
        index_params=index_params,
        max_id_len=256,
        lite_server=False,
        lite_addr=MILVUS_LITE_ADDR,
    ):
        self.uri = uri
        self.is_local = is_local_uri(uri)
        self.token = token
        self.db_name = db_name
        self.collection = collection
        self.vector_dim = vector_dim
        if lite_server and self.is_local:
            self.lite_addr = lite_addr
            self.client_uri = start_milvus_lite_server(uri, addr=lite_addr)
        else:
            self.client_uri = uri

        if self.is_local:
            index_params = {
                'index_type': 'FLAT',
                'metric_type': 'IP',
                'params': {},
            }
        self.index_params = index_params
        self.max_id_len = max_id_len
        self.client = MilvusClient(
            uri=self.client_uri,
            token=token,
            db_name=db_name,
            grpc_options=grpc_options,
        )

    def create_collection(self):
        if self.client.has_collection(self.collection):
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )
        schema.add_field(
            field_name='sample_id',
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.max_id_len,
        )
        schema.add_field(
            field_name='sample_len',
            datatype=DataType.INT32
        )
        schema.add_field(
            field_name='embedding',
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name='embedding', **self.index_params)

        self.client.create_collection(
            self.collection,
            schema=schema,
            index_params=index_params,
        )

    def drop_collection(self):
        if not self.client.has_collection(self.collection):
            return
        self.client.drop_collection(self.collection)

    def load_collection(self):
        self.client.load_collection(self.collection)

    def release_collection(self, timeout=None):
        self.client.release_collection(self.collection, timeout=timeout)

    def flush(self, timeout=None):
        self.client.flush(self.collection, timeout=timeout)

    def insert(
        self,
        sample_ids,
        sample_lengths,
        embeddings,
        batch_size=2048,
        overwrite=True,
        flush=False,
        progress_bar=False,
    ):
        size = tuple(embeddings.size())
        if size != (len(sample_ids), self.vector_dim):
            raise ValueError(
                f'Embedding size should be {(len(sample_ids), self.vector_dim)}, got {size}.'
            )
        if len(sample_ids) != len(sample_lengths):
            raise ValueError(
                'Number of sample IDs and sample lengths mismatch: '
                f'{len(sample_ids)} vs {len(sample_lengths)}.'
            )

        self.load_collection()
        it = range(0, len(sample_ids), batch_size)
        if progress_bar:
            it = tqdm(it, desc='Inserting...')
        for i in it:
            batch_ids = sample_ids[i:i + batch_size]
            batch_lengths = sample_lengths[i:i + batch_size]
            batch_embeds = embeddings[i:i + batch_size].detach().cpu().tolist()
            data = [
                {
                    'sample_id': sid,
                    'sample_len': slen,
                    'embedding': emb,
                }
                for sid, slen, emb in zip(batch_ids, batch_lengths, batch_embeds)
            ]
            if overwrite:
                self.client.upsert(self.collection, data=data)
            else:
                self.client.insert(self.collection, data=data)

        if flush:
            self.flush()

    def get(
        self,
        sample_ids,
        batch_size=2048,
        output_fields=['sample_id', 'sample_len'],
        timeout=None,
        progress_bar=False,
    ):
        self.load_collection()
        if 'sample_id' not in output_fields:
            output_fields = list(output_fields)
            output_fields.append('sample_id')

        results = []
        it = range(0, len(sample_ids), batch_size)
        if progress_bar:
            it = tqdm(it, desc='Getting...')
        for i in it:
            batch_ids = sample_ids[i:i + batch_size]
            batch_res = self.client.get(
                collection_name=self.collection,
                ids=batch_ids,
                output_fields=output_fields,
                timeout=timeout,
            )
            if batch_res:
                results.extend(batch_res)

        found_ids = {sample['sample_id'] for sample in results}
        missing_ids = [sid for sid in sample_ids if sid not in found_ids]
        return results, missing_ids

    def count(self, timeout=None):
        stats = self.client.get_collection_stats(
            collection_name=self.collection,
            timeout=timeout,
        )
        return int(stats.get('row_count', 0))

    def search(
        self,
        query_embeds,
        limit=10,
        batch_size=256,
        output_fields=['sample_id', 'sample_len'],
        filter_expr='',
        search_params=search_params,
        progress_bar=False,
    ):
        size = tuple(query_embeds.size())
        if len(size) == 1:
            if size[0] != self.vector_dim:
                raise ValueError(
                    f'Embedding size should be {(self.vector_dim,)}, got {size}.'
                )
            query_embeds = query_embeds.unsqueeze(0)
        elif len(size) == 2:
            if size[1] != self.vector_dim:
                raise ValueError(
                    f'Embedding size should be (N, {self.vector_dim}), got {size}.'
                )
        else:
            raise ValueError(
                f'Embedding size should be (N, {self.vector_dim}), got {size}.'
            )

        self.load_collection()

        results = []
        it = range(0, query_embeds.size(0), batch_size)
        if progress_bar:
            it = tqdm(it, desc='Searching...')
        for i in it:
            batch_embeds = query_embeds[i:i + batch_size].detach().cpu().tolist()
            batch_res = self.client.search(
                self.collection,
                data=batch_embeds,
                anns_field='embedding',
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
                search_params=search_params,
            )
            if batch_res:
                results.extend(batch_res)
        return results
