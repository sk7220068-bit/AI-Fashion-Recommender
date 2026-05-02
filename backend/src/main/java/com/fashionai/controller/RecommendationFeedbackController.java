package com.fashionai.controller;

import com.fashionai.model.RecommendationFeedback;
import com.fashionai.repository.RecommendationFeedbackRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/recommend-feedback")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class RecommendationFeedbackController {

    private final RecommendationFeedbackRepository feedbackRepository;

    @PostMapping
    public ResponseEntity<?> create(@RequestBody RecommendationFeedback feedback) {
        if (feedback.getUserId() == null || feedback.getUserId().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("message", "userId is required"));
        }
        if (feedback.getAction() == null || feedback.getAction().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("message", "action is required"));
        }
        RecommendationFeedback saved = feedbackRepository.save(feedback);
        return ResponseEntity.ok(saved);
    }
}
