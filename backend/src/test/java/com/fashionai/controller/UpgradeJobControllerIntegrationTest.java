package com.fashionai.controller;

import com.fashionai.model.UpgradeRenderJob;
import com.fashionai.repository.UpgradeRenderJobRepository;
import com.fashionai.service.UpgradeJobService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(UpgradeJobController.class)
class UpgradeJobControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UpgradeJobService upgradeJobService;

    @MockBean
    private UpgradeRenderJobRepository jobRepository;

    @Test
    void shouldReturnRefreshedJobPayload() throws Exception {
        UpgradeRenderJob job = UpgradeRenderJob.builder()
                .id("job_123")
                .status("running")
                .progress(62)
                .stage("inpainting")
                .upgradedImageUrl(null)
                .upgradedImageAlternatives(List.of())
                .build();

        when(upgradeJobService.refreshJob("job_123")).thenReturn(job);

        mockMvc.perform(get("/api/upgrade-jobs/job_123").accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value("job_123"))
                .andExpect(jsonPath("$.status").value("running"))
                .andExpect(jsonPath("$.progress").value(62))
                .andExpect(jsonPath("$.stage").value("inpainting"));
    }
}
