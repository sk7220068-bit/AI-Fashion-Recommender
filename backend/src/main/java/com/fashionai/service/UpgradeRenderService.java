package com.fashionai.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fashionai.model.UpgradeResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Calls the Python ML service to generate an upgraded visual preview image.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class UpgradeRenderService {

    @Value("${ml.service.base-url}")
    private String mlServiceBaseUrl;

    @Value("${ml.service.render-endpoint:/render-upgrade-preview}")
    private String renderEndpoint;

    @Value("${ml.service.timeout-seconds:30}")
    private int timeoutSeconds;

    @Value("${ml.service.api-key:test-api-key}")
    private String mlServiceApiKey;

    private final ObjectMapper objectMapper;

    public RenderResult renderUpgradePreview(byte[] imageBytes, UpgradeResult upgradeResult) {
        if (imageBytes == null || imageBytes.length == 0) {
            return RenderResult.failed();
        }

        try {
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
                    .build();

            String upgradePlanJson = objectMapper.writeValueAsString(Map.of(
                    "occasion", upgradeResult.getOccasion(),
                    "detectedItems", upgradeResult.getDetectedItems(),
                    "itemsToReplace", upgradeResult.getItemsToReplace(),
                    "itemsToAdd", upgradeResult.getItemsToAdd(),
                    "upgradeSuggestions", upgradeResult.getUpgradeSuggestions()));

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", "outfit.jpg",
                            RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                    .addFormDataPart("upgrade_plan", upgradePlanJson)
                    .build();

            Request request = new Request.Builder()
                    .url(mlServiceBaseUrl + renderEndpoint)
                    .header("X-API-Key", mlServiceApiKey)
                    .post(requestBody)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    log.warn("Upgrade render endpoint returned status {}", response.code());
                    return RenderResult.failed();
                }

                String body = response.body().string();
                Map<String, Object> result = objectMapper.readValue(body, new TypeReference<>() {});

                String upgradedImageUrl = (String) result.get("upgradedImageUrl");
                List<String> alternatives = (List<String>) result.get("upgradedImageAlternatives");

                return RenderResult.builder()
                        .success(true)
                        .upgradedImageUrl(upgradedImageUrl)
                        .upgradedImageAlternatives(alternatives)
                        .build();
            }
        } catch (Exception e) {
            log.error("Failed to call ML render service", e);
            return RenderResult.failed();
        }
    }

    @lombok.Data
    @lombok.Builder
    public static class RenderResult {
        private boolean success;
        private String upgradedImageUrl;
        private List<String> upgradedImageAlternatives;

        public static RenderResult failed() {
            return RenderResult.builder().success(false).build();
        }
    }
}
