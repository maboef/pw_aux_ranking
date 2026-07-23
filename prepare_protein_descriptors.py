
import io
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from prodec import ProteinDescriptors, Transform, TransformType


def get_kinase_domains(accessions, domain_sequences):
    print(f"Fetching domain info and sequences for {len(accessions)} entries...")
    domain_records = []
    
    for acc in accessions:
        print(acc)
    
        # 1. Fetch full sequence (FASTA)
        fasta_resp = requests.get(f"https://www.uniprot.org/uniprotkb/{acc}.fasta")
        if fasta_resp.status_code != 200:
            print(f"  Failed to fetch sequence for {acc} (status {fasta_resp.status_code})")
            continue
        try:
            full_record = SeqIO.read(io.StringIO(fasta_resp.text), "fasta")
        except Exception as e:
            print(f"  Could not parse FASTA for {acc}: {e}")
            continue
    
        # 2. Fetch feature annotations (JSON) — this is where domain coordinates live
        json_resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json")
        if json_resp.status_code != 200:
            print(f"  Failed to fetch features for {acc} (status {json_resp.status_code})")
            continue
        data = json_resp.json()
    
        kinase_domains = [
            f for f in data.get("features", [])
            if f.get("type") == "Domain" and "kinase" in f.get("description", "").lower()
        ]
    
        if not kinase_domains:
            print(f"  No annotated kinase domain found for {acc}")
            continue
    
        # 3. Slice out each kinase domain (some proteins, e.g. JAKs, have two)
        for i, dom in enumerate(kinase_domains, start=1):
            start = dom["location"]["start"]["value"]
            end = dom["location"]["end"]["value"]
            domain_seq = str(full_record.seq)[start - 1:end]  # UniProt coords are 1-based, inclusive
    
            suffix = f"_domain{i}" if len(kinase_domains) > 1 else "_domain"
            rec = SeqRecord(
                Seq(domain_seq),
                id=f"{acc}{suffix}",
                description=f"{dom.get('description', 'kinase domain')} ({start}-{end})"
            )
            domain_records.append(rec)
    
        time.sleep(0.2)  # be polite to the API
    
    # 4. Save just the domain sequences
    domain_fasta = domain_sequences
    SeqIO.write(domain_records, domain_fasta, "fasta")
    print(f"Saved {len(domain_records)} kinase domain sequences to '{domain_fasta}'")


def run_alignment_cmd(domain_sequences, aligned_sequences):
    try:
        result = subprocess.run(
            ["muscle",  "-align", f"{domain_sequences}", "-output", f"{aligned_sequences}"],
            capture_output=True,
            text=True,
            timeout=3600,
            check=True,       # raises CalledProcessError on non-zero exit
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        raise
    except FileNotFoundError:
        print(f"Command not found: {args[0]} — is it installed and on PATH?")
        raise

def generate_descriptors(aligned_sequences, out):
    records = list(SeqIO.parse("aligned_kinase_domain_sequences.fasta", "fasta"))

    df = pd.DataFrame({
        "Sequence_ID": [rec.id for rec in records],
        "Aligned_Sequence": [str(rec.seq) for rec in records],
    })
    
    pdescs = ProteinDescriptors()
    zscales = pdescs.get_descriptor('Zscale Hellberg')
    print(f"Generating features for {len(df)} sequences...")
    avg_zscale = Transform(TransformType.AVG, zscales)
    avg_descriptor_lists = df['Aligned_Sequence'].apply(lambda seq: avg_zscale.get(seq, domains=50))
    
    
    # 4. Neatly expand the lists into individual feature columns
    features_df = pd.DataFrame(list(avg_descriptor_lists))
    features_df.columns = [f"zscale_{i}" for i in range(features_df.shape[1])]
    
    # 5. Stitch it back to your original IDs
    final_df = pd.concat([df['Sequence_ID'], features_df], axis=1)
    split_cols = final_df['Sequence_ID'].str.rsplit('_', n=1, expand=True)
    final_df['accession'] = split_cols[0]
    final_df['domain'] = split_cols[1]
    
    print(f"{len(final_df['accession'].unique())} proteins, {len(final_df)} domains: keeping first")
    # keep domain or domain 1 only
    final_df = final_df[final_df['domain'].isin(['domain', 'domain1'])]
    
    zscale_cols = [c for c in final_df.columns if c.startswith('zscale_')]
    
    final_df['protein_descriptor'] = final_df[zscale_cols].values.tolist()
    
    # optionally drop the individual columns now that they're combined
    final_df = final_df.drop(columns=zscale_cols)
    final_df = final_df[['protein_descriptor', 'accession']]
    
    # 6. Save your neat ML-ready matrix
    final_df.to_pickle("kinase_features_matrix.pkl")  # no `index=` kwarg here
    print(f"Saved {len(final_df)} protein descriptors to 'kinase_features_matrix.pkl'")


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    accessions = pd.read_csv(current_dir / 'data/saifudeen_2026_raw/kinase_uniprot_target_mapping.csv')['Entry']
    prot_desc_dir = current_dir /'data/protein_descriptors/'
    domain_sequences = 'kinase_domain_sequences.fasta'
    get_kinase_domains(accessions, domain_sequences)
    
    