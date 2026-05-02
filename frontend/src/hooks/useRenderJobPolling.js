import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

export function useRenderJobPolling(initialJobId, initialStatus = 'queued', onComplete = null) {
  const [jobId, setJobId] = useState(initialJobId);
  const [status, setStatus] = useState(initialStatus);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('queued');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const timerRef = useRef(null);

  useEffect(() => {
    if (!jobId || status === 'completed' || status === 'failed') {
      return;
    }

    const poll = async () => {
      try {
        const { data } = await axios.get(`/api/upgrade-jobs/${jobId}?refresh=true`);
        setStatus(data.status);
        setProgress(data.progress || 0);
        setStage(data.stage || '');
        
        if (data.status === 'completed') {
          setResult(data.result);
          if (onComplete) onComplete(data.result);
        } else if (data.status === 'failed') {
          setError(data.error || 'Job failed');
        } else {
          timerRef.current = setTimeout(poll, 1500);
        }
      } catch (err) {
        console.error('Polling error:', err);
        setError('Error polling job status');
        setStatus('failed');
      }
    };

    timerRef.current = setTimeout(poll, 1000);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [jobId, status]);

  return { jobId, status, progress, stage, result, error, setJobId, setStatus };
}
