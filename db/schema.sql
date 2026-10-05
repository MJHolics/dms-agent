-- 졸음 감지 경보 기록 — PostgreSQL 스키마.
-- 로컬(docker-compose.db.yml의 postgres + PostgREST)과 Supabase에 같은 파일을 그대로 넣는다.
-- 브라우저는 익명 역할(anon)로 접속한다: 쓰기는 정해진 열만, 읽기는 집계 뷰만.

do $$ begin
  if not exists (select from pg_roles where rolname = 'anon') then create role anon nologin; end if;
end $$;

-- 운행 한 번 = 세션 한 건
create table if not exists dms_sessions (
  id          uuid primary key,
  vehicle     text not null check (char_length(vehicle) between 1 and 20),
  source      text not null default 'camera' check (source in ('camera', 'seed')),
  started_at  timestamptz not null default now()
);

-- 경보 한 번(시작~해제) = 이벤트 한 건. 영상·랜드마크는 저장하지 않는다.
create table if not exists dms_events (
  id          bigint generated always as identity primary key,
  session_id  uuid not null references dms_sessions (id) on delete cascade,
  ended_at    timestamptz not null default now(),          -- 서버 시각. 클라이언트가 못 정한다.
  duration_ms integer not null check (duration_ms between 0 and 600000),
  level       smallint not null check (level between 1 and 3),
  cause       text not null check (cause in ('eyes_closed', 'perclos', 'yawn', 'distracted', 'no_face', 'compound')),
  closed_sec  real check (closed_sec between 0 and 600),
  perclos     real check (perclos between 0 and 1),
  yaw         real check (yaw between -180 and 180),
  pitch       real check (pitch between -180 and 180)
);

-- 관제 화면의 세 가지 조회에 맞춘 인덱스(전후 실측은 tools/bench_db.py, reports/db_bench.json)
create index if not exists dms_events_ended_at_idx on dms_events (ended_at desc);
create index if not exists dms_events_session_idx  on dms_events (session_id, ended_at desc);
create index if not exists dms_sessions_vehicle_idx on dms_sessions (vehicle);

-- 세션당 이벤트 상한. 공개 체험 페이지라 한 세션이 테이블을 채우지 못하게 한다.
-- 익명은 테이블을 읽을 권한이 없으므로 건수 확인은 함수 소유자 권한으로 한다(security definer).
create or replace function dms_events_cap() returns trigger language plpgsql
  security definer set search_path = public as $$
begin
  if (select count(*) from dms_events where session_id = new.session_id) >= 200 then
    raise exception '세션당 이벤트 상한(200건)을 넘었습니다' using errcode = 'check_violation';
  end if;
  return new;
end $$;
drop trigger if exists dms_events_cap_trg on dms_events;
create trigger dms_events_cap_trg before insert on dms_events
  for each row execute function dms_events_cap();

-- 행 수준 보안: 익명은 넣기만 한다. 테이블을 직접 읽거나 고치거나 지우지 못한다.
alter table dms_sessions enable row level security;
alter table dms_events   enable row level security;
drop policy if exists anon_insert_session on dms_sessions;
drop policy if exists anon_insert_event   on dms_events;
create policy anon_insert_session on dms_sessions for insert to anon with check (source = 'camera');
create policy anon_insert_event   on dms_events   for insert to anon with check (true);

revoke all on dms_sessions, dms_events from anon;
grant insert (id, vehicle) on dms_sessions to anon;
grant insert (session_id, duration_ms, level, cause, closed_sec, perclos, yaw, pitch) on dms_events to anon;

-- 읽기용 뷰(뷰 소유자 권한으로 실행되므로 익명도 집계 결과는 본다)
create or replace view dms_recent_events as
  select e.id, s.vehicle, s.source, e.ended_at, e.duration_ms, e.level, e.cause, e.closed_sec, e.perclos
  from dms_events e join dms_sessions s on s.id = e.session_id
  order by e.ended_at desc
  limit 50;

create or replace view dms_vehicle_summary as
  select s.vehicle,
         count(*)                                   as alerts_24h,
         count(*) filter (where e.level = 3)        as critical_24h,
         count(*) filter (where e.cause in ('eyes_closed', 'perclos')) as drowsy_24h,
         max(e.ended_at)                            as last_alert_at
  from dms_events e join dms_sessions s on s.id = e.session_id
  where e.ended_at >= now() - interval '24 hours'
  group by s.vehicle
  order by critical_24h desc, alerts_24h desc
  limit 20;

create or replace view dms_hourly as
  select date_trunc('hour', ended_at) as hour, level, count(*) as alerts
  from dms_events
  where ended_at >= now() - interval '7 days'
  group by 1, 2
  order by 1;

grant select on dms_recent_events, dms_vehicle_summary, dms_hourly to anon;
