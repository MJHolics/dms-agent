// 경보 기록 저장소(PostgreSQL REST). 비워 두면 기록 기능이 꺼지고 나머지는 그대로 돈다.
// Supabase를 쓰면 API_URL은 https://<프로젝트>.supabase.co/rest/v1, API_KEY는 anon 공개 키다.
// (anon 키는 공개해도 되는 키다. 할 수 있는 일은 db/schema.sql의 행 수준 보안이 정한다.)
const local = ['127.0.0.1', 'localhost'].includes(location.hostname);
export const API_URL = local ? 'http://127.0.0.1:3055' : '';
export const API_KEY = '';
