package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "upgrade_render_jobs")
public class UpgradeRenderJob {

    @Id
    private String id;

    private String userId;
    private String occasion;
    private String status; // queued | running | completed | failed
    private int progress;
    private String stage;

    private String upgradedImageUrl;
    private List<String> upgradedImageAlternatives;
    private String error;
}
