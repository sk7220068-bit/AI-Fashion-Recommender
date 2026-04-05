package com.fashionai.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fashionai.model.ClothingItem;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Service that calls the Python ResNet50 feature extraction microservice to
 * obtain
 * 2048-dimensional visual feature vectors for clothing items.
 *
 * These vectors are used by:
 * - CosineSimilarityEngine (for recommendation ranking)
 * - CompatibilityScorer (for visual harmony scoring)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FeatureExtractionService {

    @Value("${ml.service.base-url}")
    private String mlServiceBaseUrl;

    @Value("${ml.service.features-endpoint}")
    private String featuresEndpoint;

    @Value("${ml.service.timeout-seconds:30}")
    private int timeoutSeconds;

    private final ObjectMapper objectMapper;

    /** Dimensionality of ResNet50's average-pooling output layer */
    private static final int FEATURE_DIM = 2048;

    /**
     * Extracts feature vectors for all detected clothing items from an image.
     * Sends the image to the Python service and enriches each ClothingItem
     * with its extracted feature vector.
     *
     * @param imageBytes raw bytes of the uploaded image
     * @param items      clothing items to enrich with feature vectors
     * @return the same list with featureVector fields populated
     */
    public List<ClothingItem> extractFeatures(byte[] imageBytes, List<ClothingItem> items) {
        log.info("Extracting features for {} clothing items", items.size());

        try {
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
                    .build();

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", "outfit.jpg",
                            RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                    .build();

            Request request = new Request.Builder()
                    .url(mlServiceBaseUrl + featuresEndpoint)
                    .post(requestBody)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    String body = response.body().string();
                    return parseFeatureResponse(body, items);
                }
            }

        } catch (IOException e) {
            log.warn("Feature extraction service unavailable: {} — generating mock features", e.getMessage());
        }

        // Fallback: generate realistic mock feature vectors for each item
        return enrichWithMockFeatures(items);
    }

    /**
     * Parses the Python service response and maps feature vectors to items by
     * category.
     *
     * Expected response format:
     * 
     * <pre>
     * {
     *   "features": {
     *     "t-shirt": [0.12, 0.34, ...],   // 2048-dim vector
     *     "jeans":   [0.56, 0.78, ...]
     *   }
     * }
     * </pre>
     */
    @SuppressWarnings("unchecked")
    private List<ClothingItem> parseFeatureResponse(String json, List<ClothingItem> items) throws IOException {
        Map<String, Object> root = objectMapper.readValue(json, new TypeReference<>() {
        });
        Map<String, List<Double>> features = (Map<String, List<Double>>) root.get("features");

        if (features != null) {
            for (ClothingItem item : items) {
                List<Double> vector = features.get(item.getCategory().toLowerCase());
                if (vector != null) {
                    item.setFeatureVector(vector);
                    log.debug("Set feature vector (dim={}) for '{}'", vector.size(), item.getCategory());
                }
            }
        }
        return items;
    }

    /**
     * Generates deterministic pseudo-random feature vectors for each item category.
     * Uses each category's hash to seed the random number generator, ensuring
     * the same category always maps to the same general feature region.
     *
     * This enables testing the entire recommendation pipeline without the ML
     * service.
     */
    private List<ClothingItem> enrichWithMockFeatures(List<ClothingItem> items) {
        // Style-based base vectors — items in the same style cluster will have
        // similar features, producing coherent (if synthetic) similarity scores
        Map<String, double[]> styleBaseVectors = generateStyleBaseVectors();

        for (ClothingItem item : items) {
            double[] baseVector = styleBaseVectors.getOrDefault(
                    item.getStyle() != null ? item.getStyle() : "casual",
                    styleBaseVectors.get("casual"));

            // Add category-specific noise on top of the style base vector
            List<Double> featureVector = new ArrayList<>(FEATURE_DIM);
            Random rand = new Random(item.getCategory().hashCode());

            for (int i = 0; i < FEATURE_DIM; i++) {
                double value = baseVector[i % baseVector.length] + (rand.nextGaussian() * 0.05);
                featureVector.add(Math.max(0, value)); // ReLU — keep non-negative
            }

            item.setFeatureVector(featureVector);
        }
        return items;
    }

    /**
     * Generates style-specific base vectors.
     * Each style cluster sits in a different region of the 2048-dim feature space.
     */
    private Map<String, double[]> generateStyleBaseVectors() {
        Map<String, double[]> bases = new HashMap<>();
        String[] styles = { "casual", "formal", "smart-casual", "sporty", "streetwear" };

        for (int s = 0; s < styles.length; s++) {
            double[] base = new double[64]; // Short base, will be tiled to 2048
            Random r = new Random(s * 1000L);
            for (int i = 0; i < base.length; i++) {
                base[i] = 0.3 + (r.nextDouble() * 0.4); // Cluster around 0.3–0.7
            }
            bases.put(styles[s], base);
        }
        return bases;
    }

    /**
     * Computes the aggregate (centroid) feature vector for a list of clothing
     * items.
     * Used to represent an entire outfit as a single vector for recommendation
     * queries.
     *
     * @param items clothing items (must have featureVector populated)
     * @return mean feature vector, or empty list if no features available
     */
    public List<Double> computeAggregateVector(List<ClothingItem> items) {
        List<List<Double>> vectors = items.stream()
                .map(ClothingItem::getFeatureVector)
                .filter(v -> v != null && !v.isEmpty())
                .toList();

        if (vectors.isEmpty())
            return List.of();

        int dim = vectors.get(0).size();
        double[] aggregate = new double[dim];

        for (List<Double> vec : vectors) {
            for (int i = 0; i < dim && i < vec.size(); i++) {
                aggregate[i] += vec.get(i);
            }
        }

        double count = vectors.size();
        List<Double> result = new ArrayList<>(dim);
        for (double v : aggregate)
            result.add(v / count);
        return result;
    }
}
