from Bio import SeqIO
import pandas as pd

def fasta_to_pandas(fasta_path):
    """
    将 FASTA 文件转换为包含 ID 和 Sequence 列的 Pandas DataFrame。
    
    参数:
        fasta_path (str): FASTA 文件的路径。
        
    返回:
        pd.DataFrame: 包含两列（ID, Sequence）的 DataFrame。
    """
    # 解析 FASTA 文件并提取 ID 和序列
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append({
            "ID": record.id,       # 序列的 ID（FASTA 头中的第一个单词）
            "Sequence": str(record.seq)  # 序列（转换为字符串）
        })
    
    # 创建 DataFrame
    df = pd.DataFrame(records)
    df["Order_ID"]=range(df.shape[0])
    return df
