"""
bench_db.py — 관제 화면 조회 3종 + 차량별 최근 기록을 인덱스 없이/있이 잰다.

    docker compose -f docker-compose.db.yml up -d
    python tools/bench_db.py            # 100만 건 적재 → 인덱스 전후 측정 → reports/db_bench.json

측정값은 EXPLAIN (ANALYZE)의 Execution Time(서버 안 실행 시간)이고, 워밍업 1회 뒤 7회의 중앙값이다.
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLES, DAYS, EVENTS = 200, 90, 1_000_000
RUNS = 7

QUERIES = {
    '최근 경보 50건(dms_recent_events)': 'select * from dms_recent_events',
    '차량별 24시간 요약(dms_vehicle_summary)': 'select * from dms_vehicle_summary',
    '7일 시간대별 건수(dms_hourly)': 'select * from dms_hourly',
    '한 세션의 최근 20건': ("select * from dms_events where session_id = "
                      "(select id from dms_sessions where source = 'seed' order by id limit 1) "
                      "order by ended_at desc limit 20"),
}
INDEXES = {
    'dms_events_ended_at_idx': 'create index dms_events_ended_at_idx on dms_events (ended_at desc)',
    'dms_events_session_idx': 'create index dms_events_session_idx on dms_events (session_id, ended_at desc)',
}


def psql(sql, *extra):
    cmd = ['docker', 'compose', '-f', os.path.join(ROOT, 'docker-compose.db.yml'), 'exec', '-T', 'db',
           'psql', '-q', '-At', '-U', 'dms', '-d', 'dms', '-v', 'ON_ERROR_STOP=1', *extra]
    r = subprocess.run(cmd, input=sql, capture_output=True, text=True, encoding='utf-8')
    if r.returncode != 0:
        sys.exit(r.stderr)
    return r.stdout.strip()


def measure():
    out = {}
    for name, q in QUERIES.items():
        times, plan = [], None
        for i in range(RUNS + 1):
            j = json.loads(psql(f'explain (analyze, format json) {q}'))[0]
            if i:
                times.append(j['Execution Time'])
            plan = j['Plan']
        nodes = []
        def walk(p):
            nodes.append(p['Node Type'] + (f"({p['Index Name']})" if 'Index Name' in p else ''))
            for c in p.get('Plans', []):
                walk(c)
        walk(plan)
        scans = [n for n in nodes if 'Scan' in n]
        out[name] = {'median_ms': round(statistics.median(times), 3), 'min_ms': round(min(times), 3),
                     'max_ms': round(max(times), 3), 'scans': scans}
    return out


def main():
    psql('truncate dms_sessions cascade')
    with open(os.path.join(ROOT, 'db', 'seed.sql'), encoding='utf-8') as f:
        psql(f.read(), '-v', f'vehicles={VEHICLES}', '-v', f'days={DAYS}', '-v', f'events={EVENTS}')
    n_s, n_e = psql('select count(*) from dms_sessions'), psql('select count(*) from dms_events')
    print(f'적재: 세션 {n_s} · 이벤트 {n_e}')

    for idx in INDEXES:
        psql(f'drop index if exists {idx}')
    psql('analyze dms_events')
    before = measure()

    build = {}
    for idx, ddl in INDEXES.items():
        psql(ddl)
        build[idx] = psql(f"select pg_size_pretty(pg_relation_size('{idx}'))")
    psql('analyze dms_events')
    after = measure()

    table_size = psql("select pg_size_pretty(pg_relation_size('dms_events'))")
    print(f'\n테이블 {table_size} · 인덱스 {build}\n')
    print(f"{'조회':<34}{'인덱스 없음':>12}{'인덱스 있음':>12}{'배':>8}")
    for name in QUERIES:
        b, a = before[name]['median_ms'], after[name]['median_ms']
        print(f'{name:<34}{b:>10.2f}ms{a:>10.2f}ms{b / a:>7.1f}x')
        print(f"    전: {before[name]['scans']}\n    후: {after[name]['scans']}")

    path = os.path.join(ROOT, 'reports', 'db_bench.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'postgres': psql('show server_version'), 'sessions': int(n_s), 'events': int(n_e),
                   'runs': RUNS, 'table_size': table_size, 'index_size': build,
                   'before': before, 'after': after}, f, ensure_ascii=False, indent=2)
    print('\n저장: reports/db_bench.json')


if __name__ == '__main__':
    main()
