package com.fashionai.service;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.Outfit;
import com.fashionai.model.UserPreference;
import com.fashionai.repository.RecommendationFeedbackRepository;
import com.fashionai.repository.UserPreferenceRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Computes user-specific personalization boosts for recommendation ranking.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PersonalizationService {

    private final UserPreferenceRepository userPreferenceRepository;
    private final RecommendationFeedbackRepository feedbackRepository;

    public Optional<UserPreference> loadPreference(String userId) {
        if (userId == null || userId.isBlank()) {
            return Optional.empty();
        }
        return userPreferenceRepository.findByUserId(userId);
    }

    public double computePersonalizationBoost(Outfit outfit, UserPreference pref, String occasion) {
        double score = 0.5;

        if (outfit.getStyle() != null && pref.getPreferredStyles().stream()
                .anyMatch(s -> s.equalsIgnoreCase(outfit.getStyle()))) {
            score += 0.15;
        }

        if (occasion != null && pref.getFrequentOccasions().stream()
                .anyMatch(o -> o.equalsIgnoreCase(occasion))) {
            score += 0.10;
        }

        Set<String> preferredColors = pref.getPreferredColors().stream()
                .map(String::toLowerCase)
                .collect(Collectors.toSet());
        if (outfit.getColorPalette() != null && !preferredColors.isEmpty()) {
            long overlap = outfit.getColorPalette().stream()
                    .map(String::toLowerCase)
                    .filter(preferredColors::contains)
                    .count();
            score += Math.min(0.15, overlap * 0.05);
        }

        Set<String> avoided = pref.getAvoidedItems().stream()
                .map(String::toLowerCase)
                .collect(Collectors.toSet());
        if (outfit.getItems() != null && !avoided.isEmpty()) {
            boolean hasAvoided = outfit.getItems().stream()
                    .map(ClothingItem::getCategory)
                    .filter(cat -> cat != null)
                    .map(String::toLowerCase)
                    .anyMatch(avoided::contains);
            if (hasAvoided) {
                score -= 0.20;
            }
        }

        if (outfit.getId() != null) {
            if (pref.getLikedOutfitIds().contains(outfit.getId())) {
                score += 0.20;
            }
            if (pref.getDislikedOutfitIds().contains(outfit.getId())) {
                score -= 0.25;
            }
        }

        return Math.max(0.0, Math.min(1.0, score));
    }

    public String buildPersonalizationReason(Outfit outfit, UserPreference pref, String occasion) {
        if (outfit.getStyle() != null && pref.getPreferredStyles().stream()
                .anyMatch(s -> s.equalsIgnoreCase(outfit.getStyle()))) {
            return "Matches your preferred style";
        }
        Set<String> preferredColors = pref.getPreferredColors().stream()
                .map(String::toLowerCase)
                .collect(Collectors.toSet());
        if (outfit.getColorPalette() != null && !preferredColors.isEmpty()) {
            boolean hasColorMatch = outfit.getColorPalette().stream()
                    .map(String::toLowerCase)
                    .anyMatch(preferredColors::contains);
            if (hasColorMatch) {
                return "Features your favorite colors";
            }
        }
        if (outfit.getId() != null && pref.getLikedOutfitIds().contains(outfit.getId())) {
            return "Based on your past likes";
        }
        return "";
    }
}
