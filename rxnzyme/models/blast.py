import os
import numpy as np
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastpCommandline, NcbimakeblastdbCommandline

def build_blast_db(fasta_path, db_name):
    print('Building BLAST database...')
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    cmd = NcbimakeblastdbCommandline(dbtype='prot', input_file=fasta_path, out=db_name)
    cmd()

def run_blast(
        query_path,
        db_name,
        max_target_seqs=100,
        evalue=0.001,
        output_path='blast_results.xml'
    ):
    print('Running BLAST...')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = NcbiblastpCommandline(
        query=query_path,
        db=db_name,
        max_target_seqs=max_target_seqs,
        evalue=evalue,
        outfmt=5,
        out=output_path
    )
    cmd()

def parse_blast_results(blast_xml, db_size, save_path=None):
    all_scores = []
    with open(blast_xml) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        for record in blast_records:
            query_scores = np.full(db_size, -100, dtype=np.float32)
            indices, scores = [], []
            
            for alignment in record.alignments:
                indices.append(int(alignment.accession))
                min_evalue = min(hsp.expect for hsp in alignment.hsps)
                # higher score means better match
                scores.append(- np.log10(min_evalue) if min_evalue > 0 else -100)
            
            query_scores[indices] = scores
            all_scores.append(query_scores)
    
    all_scores = np.stack(all_scores)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, all_scores)
    return all_scores
