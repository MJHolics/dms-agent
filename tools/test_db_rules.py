"""
test_db_rules.py — 익명 사용자가 할 수 있는 일과 없는 일을 REST로 직접 찔러 확인한다.

    docker compose -f docker-compose.db.yml up -d
    python tools/test_db_rules.py [REST 주소] [익명 키]
"""
import json
import sys
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3055'
KEY = sys.argv[2] if len(sys.argv) > 2 else ''


def call(method, path, body=None):
    headers = {'Content-Type': 'application/json', 'Prefer': 'return=minimal'}
    if KEY:
        headers.update({'apikey': KEY, 'Authorization': f'Bearer {KEY}'})
    req = urllib.request.Request(BASE + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    sid = str(uuid.uuid4())
    ev = {'session_id': sid, 'duration_ms': 2400, 'level': 3, 'cause': 'eyes_closed',
          'closed_sec': 2.4, 'perclos': 0.08, 'yaw': 1.0, 'pitch': 2.0}
    ok = lambda s: 200 <= s < 300
    cases = [
        ('세션 넣기', True, call('POST', '/dms_sessions', {'id': sid, 'vehicle': '시험 001'})),
        ('이벤트 넣기', True, call('POST', '/dms_events', ev)),
        ('집계 뷰 읽기', True, call('GET', '/dms_recent_events?limit=1')),
        ('테이블 직접 읽기', False, call('GET', '/dms_events?limit=1')),
        ('세션 테이블 직접 읽기', False, call('GET', '/dms_sessions?limit=1')),
        ('이벤트 고치기', False, call('PATCH', f'/dms_events?session_id=eq.{sid}', {'level': 1})),
        ('이벤트 지우기', False, call('DELETE', f'/dms_events?session_id=eq.{sid}')),
        ('시각 조작(ended_at 지정)', False, call('POST', '/dms_events', {**ev, 'ended_at': '2020-01-01T00:00:00Z'})),
        ('출처 위조(source=seed)', False, call('POST', '/dms_sessions', {'id': str(uuid.uuid4()), 'vehicle': 'x', 'source': 'seed'})),
        ('없는 사유', False, call('POST', '/dms_events', {**ev, 'cause': 'hacked'})),
        ('레벨 범위 밖', False, call('POST', '/dms_events', {**ev, 'level': 9})),
        ('없는 세션', False, call('POST', '/dms_events', {**ev, 'session_id': str(uuid.uuid4())})),
        ('차량 이름 21자', False, call('POST', '/dms_sessions', {'id': str(uuid.uuid4()), 'vehicle': 'x' * 21})),
    ]
    # 세션당 상한: 이미 1건 있으니 199건 더 넣으면 200건, 그다음은 막혀야 한다.
    status, _ = call('POST', '/dms_events', [ev] * 199)
    cases.append(('200건까지 넣기', True, (status, '')))
    cases.append(('201번째 이벤트', False, call('POST', '/dms_events', ev)))

    failed = 0
    for name, want, (status, text) in cases:
        good = ok(status) == want
        failed += not good
        print(f"{'통과' if good else '실패'}  {name:<26} 기대 {'허용' if want else '거부'} · HTTP {status}"
              + ('' if good else f' · {text[:120]}'))
    print(f'\n{len(cases) - failed}/{len(cases)} 통과')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
