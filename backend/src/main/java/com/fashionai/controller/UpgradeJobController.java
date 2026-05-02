package com.fashionai.controller;

import com.fashionai.model.UpgradeRenderJob;
import com.fashionai.repository.UpgradeRenderJobRepository;
import com.fashionai.service.UpgradeJobService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/upgrade-jobs")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class UpgradeJobController {

    private final UpgradeJobService upgradeJobService;
    private final UpgradeRenderJobRepository jobRepository;

    @GetMapping("/{jobId}")
    public ResponseEntity<?> get(@PathVariable String jobId,
            @RequestParam(value = "refresh", defaultValue = "true") boolean refresh) {
        UpgradeRenderJob job = refresh
                ? upgradeJobService.refreshJob(jobId)
                : jobRepository.findById(jobId).orElse(null);
        if (job == null) return ResponseEntity.notFound().build();

        return ResponseEntity.ok(Map.of(
                "jobId", job.getId(),
                "status", job.getStatus(),
                "progress", job.getProgress(),
                "stage", job.getStage() != null ? job.getStage() : "queued",
                "result", Map.of(
                        "mainImageUrl", job.getUpgradedImageUrl(),
                        "variants", job.getUpgradedImageAlternatives() != null ? job.getUpgradedImageAlternatives()
                                : java.util.List.of())));
    }
}
