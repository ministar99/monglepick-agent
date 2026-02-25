# Elasticsearch + Nori 한국어 분석기 플러그인
# §10-8: BM25 검색에 Nori 토크나이저 사용
FROM elasticsearch:8.17.0
RUN elasticsearch-plugin install analysis-nori --batch
