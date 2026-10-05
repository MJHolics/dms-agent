-- 예시 차량 기록. psql 변수: vehicles(차량 수) days(일 수) events(이벤트 수)
-- 예) psql -v vehicles=30 -v days=7 -v events=3000 -f seed.sql
alter table dms_events disable trigger dms_events_cap_trg;

insert into dms_sessions (id, vehicle, source, started_at)
select gen_random_uuid(),
       (array['화물', '지게차', '크레인'])[1 + v % 3] || ' ' || lpad(v::text, 3, '0'),
       'seed',
       now() - (d || ' days')::interval - interval '4 hours' - (random() * interval '20 hours')
from generate_series(1, :vehicles) v, generate_series(0, :days - 1) d, generate_series(1, 2) trip;

with s as (
  select id, started_at, row_number() over () as rn, count(*) over () as n from dms_sessions where source = 'seed'
), e as (
  select g, 1 + floor(random() * (select max(n) from s))::int as pick, random() as r, random() as r2
  from generate_series(1, :events) g
)
insert into dms_events (session_id, ended_at, duration_ms, level, cause, closed_sec, perclos, yaw, pitch)
select s.id,
       s.started_at + e.r2 * interval '4 hours',
       (500 + e.r2 * 6000)::int,
       case when e.r < 0.62 then 1 when e.r < 0.85 then 2 else 3 end,
       case when e.r < 0.30 then 'yawn' when e.r < 0.62 then 'distracted' when e.r < 0.85 then 'compound'
            when e.r < 0.95 then 'eyes_closed' else 'perclos' end,
       case when e.r >= 0.85 then 2 + e.r2 * 3 else 0 end,
       case when e.r >= 0.85 then 0.07 + e.r2 * 0.2 else e.r2 * 0.05 end,
       (e.r2 - 0.5) * 80, (e.r - 0.5) * 30
from e join s on s.rn = e.pick;

alter table dms_events enable trigger dms_events_cap_trg;
analyze dms_sessions; analyze dms_events;
