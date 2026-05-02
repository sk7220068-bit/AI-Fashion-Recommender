package com.fashionai.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fashionai.model.ClothingItem;
import com.fashionai.model.UpgradeRenderJob;
import com.fashionai.repository.UpgradeRenderJobRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class UpgradeJobService {

    private final UpgradeRenderJobRepository jobRepository;
    private final ObjectMapper objectMapper;

    @Value("${ml.service.base-url}")
    private String mlServiceBaseUrl;

    @Value("${ml.service.generate-endpoint:/generate-upgrade}")
    private String generateEndpoint;

    @Value("${ml.service.jobs-endpoint:/jobs}")
    private String jobsEndpoint;

    @Value("${ml.service.api-key:test-api-key}")
    private String mlServiceApiKey;

    public UpgradeRenderJob createJob(String userId, String occasion, byte[] imageBytes,
            List<ClothingItem> detectedItems, List<String> itemsToReplace, List<String> itemsToAdd) {
        UpgradeRenderJob job = UpgradeRenderJob.builder()
                .userId(userId)
                .occasion(occasion)
                .status("queued")
                .progress(0)
                .stage("queued")
                .build();
        job = jobRepository.save(job);

        try {
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .build();

            String plan = objectMapper.writeValueAsString(Map.of(
                    "jobId", job.getId(),
                    "userId", userId != null ? userId : "",
                    "occasion", occasion,
                    "detectedItems", detectedItems,
                    "itemsToReplace", itemsToReplace,
                    "itemsToAdd", itemsToAdd));

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", "outfit.jpg",
                            RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                    .addFormDataPart("upgrade_plan", plan)
                    .build();

            Request request = new Request.Builder()
                    .url(mlServiceBaseUrl + generateEndpoint)
                    .header("X-API-Key", mlServiceApiKey)
                    .post(requestBody)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    job.setStatus("failed");
                    job.setError("ML job creation failed with status " + response.code());
                    return jobRepository.save(job);
                }
                Map<String, Object> payload = objectMapper.readValue(response.body().string(), new TypeReference<>() {});
                String returnedJobId = (String) payload.getOrDefault("jobId", job.getId());
                if (!returnedJobId.equals(job.getId())) {
                    log.info("ML service returned different job id '{}', backend uses '{}'.", returnedJobId, job.getId());
                }
                return job;
            }
        } catch (Exception e) {
            log.error("Failed to create ML job", e);
            job.setStatus("failed");
            job.setError("Internal backend error: " + e.getMessage());
            return jobRepository.save(job);
        }
    }

    public UpgradeRenderJob refreshJob(String jobId) {
        UpgradeRenderJob job = jobRepository.findById(jobId).orElse(null);
        if (job == null) return null;
        if ("completed".equals(job.getStatus()) || "failed".equals(job.getStatus())) {
            return job;
        }

        try {
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(5, TimeUnit.SECONDS)
                    .readTimeout(10, TimeUnit.SECONDS)
                    .build();

            Request request = new Request.Builder()
                    .url(mlServiceBaseUrl + jobsEndpoint + "/" + jobId)
                    .header("X-API-Key", mlServiceApiKey)
                    .get()
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    Map<String, Object> body = objectMapper.readValue(response.body().string(), new TypeReference<>() {});
                    String status = (String) body.get("status");
                    if (status != null) job.setStatus(status);
                    
                    if (body.containsKey("progress")) {
                        job.setProgress(((Number) body.get("progress")).intValue());
                    }
                    if (body.containsKey("stage")) {
                        job.setStage((String) body.get("stage"));
                    }
                    if (body.containsKey("result") && body.get("result") instanceof Map) {
                        Map<String, Object> result = (Map<String, Object>) body.get("result");
                        job.setUpgradedImageUrl((String) result.get("mainImageUrl"));
                        job.setUpgradedImageAlternatives((List<String>) result.get("variants"));
                    }
                    if (body.containsKey("error")) {
                        job.setError((String) body.get("error"));
                    }
                    return jobRepository.save(job);
                } else if (response.code() == 404) {
                    job.setStatus("failed");
                    job.setError("Job not found on ML service");
                    return jobRepository.save(job);
                }
            }
        } catch (Exception e) {
            log.error("Failed to refresh ML job status", e);
        }

        return job;
    }
}
