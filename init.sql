-- 몽글픽 MySQL 초기화 스크립트
-- Spring Boot 백엔드가 관리하는 테이블의 AI 파이프라인 참조용 스키마

CREATE DATABASE IF NOT EXISTS monglepick
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE monglepick;

-- AI 추천 내역 로그 (AI Agent가 직접 기록)
CREATE TABLE IF NOT EXISTS recommendations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    movie_id INT NOT NULL COMMENT 'TMDB ID',
    session_id VARCHAR(36) NOT NULL,
    reason TEXT NOT NULL COMMENT '추천 이유 (AI 생성)',
    score FLOAT NOT NULL COMMENT '최종 추천 점수',
    cf_score FLOAT DEFAULT NULL,
    cbf_score FLOAT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
