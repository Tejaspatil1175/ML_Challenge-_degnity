import duckdb
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

con = duckdb.connect()
con.execute('SET threads=4')
con.execute('SET memory_limit="6GB"')

# Take a random sample of 2,000 S1 entities
print("Sampling 2,000 S1 entities to evaluate candidate density...")
con.execute("""
CREATE TABLE s1_sample AS 
SELECT 
    entity_id,
    clean_name,
    clean_address,
    clean_country,
    name_prefix_4,
    clean_zipcode,
    clean_city,
    regexp_extract(clean_address, '([0-9]+[a-z0-9\-\/]*)', 1) as addr_num,
    list_sort(list_filter(string_split(regexp_replace(clean_name, '(private|limited|incorporated|corp|corporation|llc|inc|ltd|co)', '', 'g'), ' '), x -> length(x) >= 3))[1] as name_tok1,
    list_sort(list_filter(string_split(regexp_replace(clean_name, '(private|limited|incorporated|corp|corporation|llc|inc|ltd|co)', '', 'g'), ' '), x -> length(x) >= 3))[2] as name_tok2,
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[1] as addr_tok1,
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[2] as addr_tok2
FROM read_parquet('data_cache/processed/train_s1_norm.parquet')
USING SAMPLE 2000;
""")

print("Loading full S2 and S3 for density join...")
# S2 and S3 full tables
con.execute("""
CREATE VIEW s23_full AS 
SELECT 
    entity_id,
    clean_name,
    clean_address,
    clean_country,
    name_prefix_4,
    clean_zipcode,
    clean_city,
    regexp_extract(clean_address, '([0-9]+[a-z0-9\-\/]*)', 1) as addr_num,
    list_sort(list_filter(string_split(regexp_replace(clean_name, '(private|limited|incorporated|corp|corporation|llc|inc|ltd|co)', '', 'g'), ' '), x -> length(x) >= 3))[1] as name_tok1,
    list_sort(list_filter(string_split(regexp_replace(clean_name, '(private|limited|incorporated|corp|corporation|llc|inc|ltd|co)', '', 'g'), ' '), x -> length(x) >= 3))[2] as name_tok2,
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[1] as addr_tok1,
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[2] as addr_tok2
FROM read_parquet(['data_cache/processed/train_s2_norm.parquet', 'data_cache/processed/train_s3_norm.parquet']);
""")

rules_cond = """
(s1.clean_name = s2.clean_name AND s1.clean_name != '')
OR (s1.name_prefix_4 = s2.name_prefix_4 AND length(s1.name_prefix_4)>=3 AND s1.clean_city = s2.clean_city AND s1.clean_city != '')
OR (s1.name_prefix_4 = s2.name_prefix_4 AND length(s1.name_prefix_4)>=3 AND s1.clean_zipcode = s2.clean_zipcode AND s1.clean_zipcode != '')
OR (s1.name_tok1 = s2.name_tok1 AND s1.name_tok2 = s2.name_tok2 AND s1.name_tok1 IS NOT NULL AND s1.name_tok2 IS NOT NULL)
OR (s1.addr_num = s2.addr_num AND length(s1.addr_num) >= 2 AND s1.clean_city = s2.clean_city AND s1.clean_city != '')
OR (s1.addr_num = s2.addr_num AND length(s1.addr_num) >= 2 AND s1.clean_zipcode = s2.clean_zipcode AND s1.clean_zipcode != '')
OR (substring(s1.clean_address, 1, 15) = substring(s2.clean_address, 1, 15) AND length(s1.clean_address) >= 12)
OR (s1.addr_tok1 = s2.addr_tok1 AND s1.addr_tok2 = s2.addr_tok2 AND s1.addr_tok1 IS NOT NULL AND s1.addr_tok2 IS NOT NULL)
"""

print("Running join on 2,000 S1 sample...")
cands = con.execute(f"""
SELECT 
    SUM(cand_count) as total_cands,
    COUNT(DISTINCT entity_id) as matched_s1,
    AVG(cand_count) as avg_cands,
    MAX(cand_count) as max_cands
FROM (
    SELECT s1.entity_id, COUNT(*) as cand_count
    FROM s1_sample s1
    JOIN s23_full s2 ON ({rules_cond})
    GROUP BY s1.entity_id
);
""").fetchone()

print(f"Results on 2,000 sample:")
print(f"  Total candidate pairs: {cands[0]:,}")
print(f"  S1 entities with cands: {cands[1]:,}/2,000 ({cands[1]/20:.1f}%)")
print(f"  Avg candidates per entity: {cands[2]:.2f}")
print(f"  Max candidates on an entity: {cands[3]}")
