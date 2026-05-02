import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 20,
  iterations: 20,
};

export default function () {
  const res = http.get(`${__ENV.BASE_URL || 'http://localhost:8080'}/api/upgrade-jobs/job_smoke?refresh=true`);
  check(res, {
    'status is 200 or 404': (r) => r.status === 200 || r.status === 404,
  });
  sleep(0.2);
