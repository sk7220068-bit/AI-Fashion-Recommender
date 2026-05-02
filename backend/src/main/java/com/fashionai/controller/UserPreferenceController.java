package com.fashionai.controller;

import com.fashionai.model.UserPreference;
import com.fashionai.repository.UserPreferenceRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/user-preferences")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class UserPreferenceController {

    private final UserPreferenceRepository userPreferenceRepository;

    @PostMapping
    public ResponseEntity<?> upsert(@RequestBody UserPreference preference) {
        if (preference.getUserId() == null || preference.getUserId().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("message", "userId is required"));
        }

        UserPreference existing = userPreferenceRepository.findByUserId(preference.getUserId())
                .orElse(null);
        if (existing != null) {
            preference.setId(existing.getId());
        }

        UserPreference saved = userPreferenceRepository.save(preference);
        return ResponseEntity.ok(saved);
    }

    @GetMapping("/{userId}")
    public ResponseEntity<?> get(@PathVariable String userId) {
        return userPreferenceRepository.findByUserId(userId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }
}
