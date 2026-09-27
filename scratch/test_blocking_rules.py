import duckdb
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

con = duckdb.connect()
con.execute('SET threads=4')
con.execute('SET memory_limit="6GB"')

print("Loading 50,000 true pairs...")
con.execute("""
CREATE TABLE test_gt AS 
SELECT source1_entity_id, candidate_entity_id
FROM (
    SELECT 
        source1_entity_id, 
        unnest(string_split(matched_entity_ids, ',')) as candidate_entity_id
    FROM read_csv_auto('6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv', delim='\t')
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    LIMIT 50000
);
""")
n_true = con.execute("SELECT COUNT(*) FROM test_gt").fetchone()[0]

con.execute("""
CREATE TABLE s1_sub AS 
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
    -- Address significant tokens (sorted, >= 5 chars, excluding road, street, floor, apartment, etc)
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[1] as addr_tok1,
    list_sort(list_filter(string_split(regexp_replace(clean_address, '(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot)', '', 'g'), ' '), x -> length(x) >= 5))[2] as addr_tok2
FROM read_parquet('data_cache/processed/train_s1_norm.parquet')
WHERE entity_id IN (SELECT source1_entity_id FROM test_gt);
""")

con.execute("""
CREATE TABLE s23_sub AS 
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
FROM read_parquet(['data_cache/processed/train_s2_norm.parquet', 'data_cache/processed/train_s3_norm.parquet'])
WHERE entity_id IN (SELECT candidate_entity_id FROM test_gt);
""")

# Test Address Token Match
r_addr_tok = con.execute("""
SELECT COUNT(DISTINCT g.source1_entity_id || '_' || g.candidate_entity_id)
FROM test_gt g
JOIN s1_sub s1 ON g.source1_entity_id = s1.entity_id
JOIN s23_sub s2 ON g.candidate_entity_id = s2.entity_id
WHERE s1.addr_tok1 = s2.addr_tok1 AND s1.addr_tok2 = s2.addr_tok2 AND s1.addr_tok1 IS NOT NULL AND s1.addr_tok2 IS NOT NULL;
""").fetchone()[0]
print(f"Address Significant Tokens 1 & 2 match: {r_addr_tok}/{n_true} ({r_addr_tok/n_true*100:.2f}%)")

# Test ALL COMBINED with Address Tokens:
rules_cond = """
(s1.clean_name = s2.clean_name AND s1.clean_name != '')
OR (s1.name_prefix_4 = s2.name_prefix_4 AND length(s1.name_prefix_4)>=3 AND s1.clean_city = s2.clean_city AND s1.clean_city != '')
OR (s1.name_prefix_4 = s2.name_prefix_4 AND length(s1.name_prefix_4)>=3 AND s1.clean_zipcode = s2.clean_zipcode AND s1.clean_zipcode != '')
OR (s1.name_tok1 = s2.name_tok1 AND s1.name_tok2 = s2.name_tok2 AND s1.name_tok1 IS NOT NULL AND s1.name_tok2 IS NOT NULL)
OR (s1.addr_num = s2.addr_num AND length(s1.addr_num) >= 2 AND s1.clean_city = s2.clean_city AND s1.clean_city != '')
OR (s1.addr_num = s2.addr_num AND length(s1.addr_num) >= 2 AND s1.clean_zipcode = s2.clean_zipcode AND s1.clean_zipcode != '')
OR (s1.name_prefix_4 = s2.name_prefix_4 AND length(s1.name_prefix_4) >= 4 AND (s1.clean_country = s2.clean_country OR s1.clean_country='' OR s2.clean_country=''))
OR (substring(s1.clean_address, 1, 15) = substring(s2.clean_address, 1, 15) AND length(s1.clean_address) >= 12)
OR (s1.addr_tok1 = s2.addr_tok1 AND s1.addr_tok2 = s2.addr_tok2 AND s1.addr_tok1 IS NOT NULL AND s1.addr_tok2 IS NOT NULL)
"""

tot_cnt = con.execute(f"""
SELECT COUNT(DISTINCT g.source1_entity_id || '_' || g.candidate_entity_id)
FROM test_gt g
JOIN s1_sub s1 ON g.source1_entity_id = s1.entity_id
JOIN s23_sub s2 ON g.candidate_entity_id = s2.entity_id
WHERE {rules_cond};
""").fetchone()[0]

print("=" * 60)
print(f"NEW OVERALL BLOCKING RECALL: {tot_cnt}/{n_true} ({tot_cnt/n_true*100:.2f}%)")
print("=" * 60)
