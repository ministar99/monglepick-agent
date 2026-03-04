-- ============================================================
-- 몽글픽 MySQL 전체 스키마 (15개 테이블)
-- ============================================================
--
-- 설계서 §4-8 기준 + Qdrant/Neo4j/ES와 동기화되는 영화 경량 참조 테이블.
-- Spring Boot 백엔드와 AI Agent 양쪽에서 참조한다.
--
-- 테이블 목록:
--   1. movies             — 영화 경량 참조 (Qdrant 미러링)
--   2. users              — 사용자 기본 정보
--   3. user_preferences   — 사용자 취향 (JSON)
--   4. watch_history      — 시청 이력 + 평점
--   5. user_wishlist      — 찜 목록
--   6. recommendation_log — 추천 이력 로그
--   7. recommendation_feedback — 추천 피드백
--   8. movie_mentions     — 커뮤니티 영화 언급 집계
--   9. user_achievements  — 사용자 업적 (도장깨기)
--  10. toxicity_log       — 비속어 검출 로그
--  11. chat_session_archive — 대화 세션 아카이브
--  12. posts              — 커뮤니티 게시글
--  13. reviews            — 영화 리뷰
--  14. roadmap_courses    — 도장깨기 코스
--  15. quiz_attempts      — 퀴즈 도전 기록
-- ============================================================

CREATE DATABASE IF NOT EXISTS monglepick
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE monglepick;


-- ============================================================
-- 1. movies — 영화 경량 참조 테이블
-- ============================================================
-- Qdrant/Neo4j/ES에 상세 데이터가 있고, MySQL에는 Spring Boot가
-- 참조할 경량 컬럼만 저장한다. Qdrant scroll → 배치 INSERT로 동기화.
--
-- movie_id: TMDB ID(숫자), KOBIS 코드(영문 포함), KMDb ID(숫자) 등
-- 다양한 소스의 ID가 공존하므로 VARCHAR(50)으로 통일.
-- ============================================================
CREATE TABLE IF NOT EXISTS movies (
    movie_id        VARCHAR(50)  NOT NULL PRIMARY KEY COMMENT '영화 ID (TMDB/KOBIS/KMDb)',
    title           VARCHAR(500) NOT NULL             COMMENT '한국어 제목',
    title_en        VARCHAR(500) DEFAULT NULL          COMMENT '영어 제목',
    poster_path     VARCHAR(500) DEFAULT NULL          COMMENT 'TMDB 포스터 경로',
    backdrop_path   VARCHAR(500) DEFAULT NULL          COMMENT 'TMDB 배경 이미지 경로',
    release_year    INT          DEFAULT NULL          COMMENT '개봉 연도',
    runtime         INT          DEFAULT NULL          COMMENT '상영 시간 (분)',
    rating          FLOAT        DEFAULT NULL          COMMENT '평균 평점 (0~10)',
    vote_count      INT          DEFAULT NULL          COMMENT '투표 수',
    popularity_score FLOAT       DEFAULT NULL          COMMENT 'TMDB 인기도 점수',
    genres          JSON         DEFAULT NULL          COMMENT '장르 목록 ["액션","드라마"]',
    director        VARCHAR(200) DEFAULT NULL          COMMENT '감독 이름',
    cast            JSON         DEFAULT NULL          COMMENT '주연 배우 목록 ["배우1","배우2"]',
    certification   VARCHAR(50)  DEFAULT NULL          COMMENT '관람등급 (전체관람가, 12세 등)',
    trailer_url     VARCHAR(500) DEFAULT NULL          COMMENT 'YouTube 트레일러 URL',
    overview        TEXT         DEFAULT NULL          COMMENT '줄거리',
    tagline         VARCHAR(500) DEFAULT NULL          COMMENT '태그라인',
    imdb_id         VARCHAR(20)  DEFAULT NULL          COMMENT 'IMDb ID (tt로 시작)',
    original_language VARCHAR(10) DEFAULT NULL         COMMENT '원본 언어 코드 (en, ko 등)',
    collection_name VARCHAR(200) DEFAULT NULL          COMMENT '프랜차이즈/컬렉션 이름',
    -- KOBIS 보강 컬럼
    kobis_movie_cd  VARCHAR(20)  DEFAULT NULL          COMMENT 'KOBIS 영화 코드',
    sales_acc       BIGINT       DEFAULT NULL          COMMENT '누적 매출액 (KRW)',
    audience_count  BIGINT       DEFAULT NULL          COMMENT '관객수',
    screen_count    INT          DEFAULT NULL          COMMENT '최대 상영 스크린 수',
    kobis_watch_grade VARCHAR(50) DEFAULT NULL         COMMENT 'KOBIS 관람등급',
    kobis_open_dt   VARCHAR(10)  DEFAULT NULL          COMMENT 'KOBIS 개봉일 (YYYYMMDD)',
    -- KMDb 보강 컬럼
    kmdb_id         VARCHAR(50)  DEFAULT NULL          COMMENT 'KMDb 영화 ID',
    awards          TEXT         DEFAULT NULL          COMMENT '수상 내역',
    filming_location TEXT        DEFAULT NULL          COMMENT '촬영 장소',
    -- 데이터 출처 추적
    source          VARCHAR(20)  DEFAULT NULL          COMMENT '데이터 출처 (tmdb/kaggle/kobis/kmdb)',
    -- 타임스탬프
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    -- 인덱스
    INDEX idx_movies_title (title(100)),
    INDEX idx_movies_release_year (release_year),
    INDEX idx_movies_rating (rating),
    INDEX idx_movies_popularity (popularity_score),
    INDEX idx_movies_director (director),
    INDEX idx_movies_source (source),
    INDEX idx_movies_imdb_id (imdb_id),
    INDEX idx_movies_kobis_cd (kobis_movie_cd),
    INDEX idx_movies_kmdb_id (kmdb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 2. users — 사용자 기본 정보
-- ============================================================
-- Spring Boot 회원가입 시 생성. AI Agent는 읽기 전용.
-- Kaggle 시드 유저는 user_id = 'kaggle_{userId}' 형태로 구분.
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(50)  NOT NULL PRIMARY KEY COMMENT '사용자 ID',
    nickname        VARCHAR(100) DEFAULT NULL          COMMENT '닉네임',
    email           VARCHAR(200) DEFAULT NULL          COMMENT '이메일',
    profile_image   VARCHAR(500) DEFAULT NULL          COMMENT '프로필 이미지 URL',
    age_group       VARCHAR(10)  DEFAULT NULL          COMMENT '연령대 (10대, 20대 등)',
    gender          VARCHAR(10)  DEFAULT NULL          COMMENT '성별 (M/F/O)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 3. user_preferences — 사용자 취향 프로필
-- ============================================================
-- §6-4 preference_refiner에서 추출한 선호 조건을 누적 저장.
-- JSON 필드로 유연한 스키마 지원.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_preferences (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    preferred_genres JSON        DEFAULT NULL           COMMENT '선호 장르 ["액션","SF"]',
    preferred_moods  JSON        DEFAULT NULL           COMMENT '선호 무드 ["스릴","감동"]',
    preferred_directors JSON     DEFAULT NULL           COMMENT '선호 감독 ["봉준호"]',
    preferred_actors JSON        DEFAULT NULL           COMMENT '선호 배우 ["송강호"]',
    preferred_eras   JSON        DEFAULT NULL           COMMENT '선호 시대 ["2020s"]',
    excluded_genres  JSON        DEFAULT NULL           COMMENT '제외 장르 ["호러"]',
    preferred_platforms JSON     DEFAULT NULL           COMMENT '선호 OTT ["넷플릭스"]',
    preferred_certification VARCHAR(50) DEFAULT NULL    COMMENT '선호 관람등급',
    -- 누적 대화에서 학습된 추가 선호 (자유 형식)
    extra_preferences JSON      DEFAULT NULL           COMMENT '추가 선호 조건 (키-값 자유 형식)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_preferences (user_id),
    CONSTRAINT fk_user_preferences_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 4. watch_history — 시청 이력 + 평점
-- ============================================================
-- 사용자가 본 영화와 평점을 기록한다.
-- Kaggle ratings.csv (26M행) 시드 데이터를 여기에 적재.
-- ============================================================
CREATE TABLE IF NOT EXISTS watch_history (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    rating          FLOAT        DEFAULT NULL           COMMENT '사용자 평점 (0.5~5.0)',
    watched_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '시청 일시',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_watch_user (user_id),
    INDEX idx_watch_movie (movie_id),
    INDEX idx_watch_user_movie (user_id, movie_id),
    INDEX idx_watch_watched_at (watched_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 5. user_wishlist — 찜 목록
-- ============================================================
-- 사용자가 '나중에 볼 영화'로 찜한 영화 목록.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_wishlist (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_wishlist_user_movie (user_id, movie_id),
    INDEX idx_wishlist_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 6. recommendation_log — 추천 이력 로그
-- ============================================================
-- AI Agent가 추천을 생성할 때마다 기록한다.
-- 기존 recommendations 테이블을 대체.
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendation_log (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    session_id      VARCHAR(36)  NOT NULL              COMMENT '대화 세션 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '추천된 영화 ID',
    reason          TEXT         NOT NULL               COMMENT '추천 이유 (AI 생성)',
    score           FLOAT        NOT NULL               COMMENT '최종 추천 점수',
    cf_score        FLOAT        DEFAULT NULL           COMMENT 'CF 점수',
    cbf_score       FLOAT        DEFAULT NULL           COMMENT 'CBF 점수',
    hybrid_score    FLOAT        DEFAULT NULL           COMMENT '하이브리드 합산 점수',
    genre_match     FLOAT        DEFAULT NULL           COMMENT '장르 일치도',
    mood_match      FLOAT        DEFAULT NULL           COMMENT '무드 일치도',
    rank_position   INT          DEFAULT NULL           COMMENT '추천 순위 (1~5)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reclog_user (user_id),
    INDEX idx_reclog_session (session_id),
    INDEX idx_reclog_movie (movie_id),
    INDEX idx_reclog_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 7. recommendation_feedback — 추천 피드백
-- ============================================================
-- 사용자가 추천 결과에 대해 남긴 피드백 (좋아요/싫어요/이미 봤어요).
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    recommendation_id BIGINT     NOT NULL               COMMENT '추천 로그 ID',
    feedback_type   ENUM('like','dislike','watched','not_interested')
                                 NOT NULL               COMMENT '피드백 유형',
    comment         TEXT         DEFAULT NULL           COMMENT '사용자 코멘트',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_feedback_user_rec (user_id, recommendation_id),
    INDEX idx_feedback_rec (recommendation_id),
    CONSTRAINT fk_feedback_rec FOREIGN KEY (recommendation_id)
        REFERENCES recommendation_log(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 8. movie_mentions — 커뮤니티 영화 언급 집계
-- ============================================================
-- §8 콘텐츠 분석 에이전트가 커뮤니티에서 영화 언급을 수집·집계.
-- 기간별 버즈량 추적에 사용.
-- ============================================================
CREATE TABLE IF NOT EXISTS movie_mentions (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    source          VARCHAR(50)  NOT NULL              COMMENT '소스 (reddit, twitter, naver 등)',
    mention_count   INT          DEFAULT 0              COMMENT '언급 횟수',
    sentiment_avg   FLOAT        DEFAULT NULL           COMMENT '평균 감성 점수 (-1~1)',
    period_start    DATE         NOT NULL               COMMENT '집계 기간 시작',
    period_end      DATE         NOT NULL               COMMENT '집계 기간 종료',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_mention_movie_source_period (movie_id, source, period_start),
    INDEX idx_mention_movie (movie_id),
    INDEX idx_mention_period (period_start, period_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 9. user_achievements — 사용자 업적 (도장깨기)
-- ============================================================
-- §9 로드맵 에이전트의 도장깨기 코스 달성 기록.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_achievements (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    achievement_type VARCHAR(50) NOT NULL               COMMENT '업적 유형 (course_complete, quiz_pass 등)',
    achievement_key VARCHAR(100) NOT NULL               COMMENT '업적 키 (코스 ID, 퀴즈 ID 등)',
    achieved_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '달성 일시',
    metadata        JSON         DEFAULT NULL           COMMENT '업적 메타데이터 (점수, 순위 등)',
    UNIQUE KEY uk_achievement (user_id, achievement_type, achievement_key),
    INDEX idx_achievement_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 10. toxicity_log — 비속어 검출 로그
-- ============================================================
-- §8 콘텐츠 분석 에이전트가 검출한 비속어/유해 표현 로그.
-- ============================================================
CREATE TABLE IF NOT EXISTS toxicity_log (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  DEFAULT NULL           COMMENT '사용자 ID (익명 가능)',
    session_id      VARCHAR(36)  DEFAULT NULL           COMMENT '대화 세션 ID',
    input_text      TEXT         NOT NULL               COMMENT '원본 입력 텍스트',
    toxicity_score  FLOAT        NOT NULL               COMMENT '유해도 점수 (0~1)',
    toxicity_type   VARCHAR(50)  DEFAULT NULL           COMMENT '유해 유형 (profanity, hate 등)',
    action_taken    VARCHAR(50)  DEFAULT 'flagged'      COMMENT '조치 (flagged, blocked, warned)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_toxicity_user (user_id),
    INDEX idx_toxicity_session (session_id),
    INDEX idx_toxicity_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 11. chat_session_archive — 대화 세션 아카이브
-- ============================================================
-- Redis에서 TTL 만료 전 대화 세션을 MySQL에 아카이브.
-- 장기 분석 및 학습 데이터로 활용.
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_session_archive (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    session_id      VARCHAR(36)  NOT NULL              COMMENT '대화 세션 ID',
    messages        JSON         NOT NULL               COMMENT '전체 대화 메시지 배열',
    turn_count      INT          DEFAULT 0              COMMENT '대화 턴 수',
    intent_summary  JSON         DEFAULT NULL           COMMENT '의도 분류 요약',
    started_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '대화 시작 시각',
    ended_at        TIMESTAMP    DEFAULT NULL           COMMENT '대화 종료 시각',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_session (session_id),
    INDEX idx_archive_user (user_id),
    INDEX idx_archive_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 12. posts — 커뮤니티 게시글
-- ============================================================
-- Spring Boot 커뮤니티 기능. AI Agent는 읽기 전용 분석.
-- ============================================================
CREATE TABLE IF NOT EXISTS posts (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    title           VARCHAR(300) NOT NULL               COMMENT '게시글 제목',
    content         TEXT         NOT NULL               COMMENT '게시글 본문',
    category        VARCHAR(50)  DEFAULT 'general'     COMMENT '카테고리 (general, review, discussion)',
    movie_id        VARCHAR(50)  DEFAULT NULL           COMMENT '관련 영화 ID (없을 수 있음)',
    like_count      INT          DEFAULT 0              COMMENT '좋아요 수',
    comment_count   INT          DEFAULT 0              COMMENT '댓글 수',
    view_count      INT          DEFAULT 0              COMMENT '조회 수',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_posts_user (user_id),
    INDEX idx_posts_movie (movie_id),
    INDEX idx_posts_category (category),
    INDEX idx_posts_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 13. reviews — 영화 리뷰
-- ============================================================
-- 사용자가 작성한 영화 리뷰. 평점 + 텍스트.
-- ============================================================
CREATE TABLE IF NOT EXISTS reviews (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    rating          FLOAT        NOT NULL               COMMENT '평점 (0.5~5.0)',
    content         TEXT         DEFAULT NULL           COMMENT '리뷰 본문',
    spoiler         BOOLEAN      DEFAULT FALSE          COMMENT '스포일러 포함 여부',
    like_count      INT          DEFAULT 0              COMMENT '좋아요 수',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_review_user_movie (user_id, movie_id),
    INDEX idx_reviews_movie (movie_id),
    INDEX idx_reviews_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 14. roadmap_courses — 도장깨기 코스
-- ============================================================
-- §9 로드맵 에이전트가 생성하는 영화 도장깨기 코스.
-- 각 코스는 테마별 영화 목록 + 순서를 포함.
-- ============================================================
CREATE TABLE IF NOT EXISTS roadmap_courses (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    course_id       VARCHAR(50)  NOT NULL              COMMENT '코스 고유 ID',
    title           VARCHAR(300) NOT NULL               COMMENT '코스 제목 ("봉준호 감독 마스터 코스")',
    description     TEXT         DEFAULT NULL           COMMENT '코스 설명',
    theme           VARCHAR(100) DEFAULT NULL           COMMENT '코스 테마 (감독, 장르, 시대 등)',
    movie_ids       JSON         NOT NULL               COMMENT '코스에 포함된 영화 ID 배열 (순서 보장)',
    movie_count     INT          NOT NULL               COMMENT '코스 내 영화 수',
    difficulty      ENUM('beginner','intermediate','advanced')
                                 DEFAULT 'beginner'    COMMENT '난이도',
    quiz_enabled    BOOLEAN      DEFAULT FALSE          COMMENT '퀴즈 포함 여부',
    created_by      VARCHAR(50)  DEFAULT 'ai_agent'    COMMENT '생성자 (ai_agent 또는 user_id)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_course_id (course_id),
    INDEX idx_course_theme (theme),
    INDEX idx_course_difficulty (difficulty)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 15. quiz_attempts — 퀴즈 도전 기록
-- ============================================================
-- 도장깨기 코스의 퀴즈 도전 기록.
-- ============================================================
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    course_id       VARCHAR(50)  NOT NULL              COMMENT '코스 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '퀴즈 대상 영화 ID',
    question        TEXT         NOT NULL               COMMENT '퀴즈 질문',
    user_answer     TEXT         NOT NULL               COMMENT '사용자 답변',
    correct_answer  TEXT         NOT NULL               COMMENT '정답',
    is_correct      BOOLEAN      NOT NULL               COMMENT '정답 여부',
    score           INT          DEFAULT 0              COMMENT '획득 점수',
    attempted_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_quiz_user (user_id),
    INDEX idx_quiz_course (course_id),
    INDEX idx_quiz_user_course (user_id, course_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
