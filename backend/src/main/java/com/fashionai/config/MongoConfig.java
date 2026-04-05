package com.fashionai.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.config.EnableMongoAuditing;

/**
 * MongoDB and Jackson configuration.
 *
 * Enables:
 *   - MongoDB auditing (auto-populate @CreatedDate fields)
 *   - Jackson ISO-8601 date formatting (not Unix timestamps)
 */
@Configuration
@EnableMongoAuditing
public class MongoConfig {

    /**
     * Provides a globally configured ObjectMapper.
     * Handles Java 8 date/time types (Instant, LocalDate) correctly.
     */
    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        mapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        return mapper;
    }
}
