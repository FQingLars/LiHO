//! Configuration loading: `.env` file + process environment.
//! Secrets are never hardcoded and never logged — only their metadata.

use std::collections::HashMap;

use log::info;

/// Default role instruction sent with the first question of the session.
pub const DEFAULT_SYSTEM_PROMPT: &str = "Ты — ассистент на техническом интервью. Тебе пересылают распознанный текст вопроса интервьюера. Отвечай кратко и продукто-ориентированно, как будто мы заявляем о преимуществе решения для продукта. Первый ответ — сразу суть, без вступлений. Язык ответа — язык вопроса.";

#[derive(Debug, Clone)]
pub struct Config {
    pub deepseek_token: String,
    /// Path to the local T-one ONNX acoustic model.
    pub stt_model: std::path::PathBuf,
    /// Role instruction prepended to the first question of a session.
    pub system_prompt: String,
}

impl Config {
    /// Load config: read `.env` if present, then process environment.
    pub fn load() -> Result<Config, String> {
        match dotenvy::dotenv() {
            Ok(path) => info!("Loaded env file: {}", path.display()),
            Err(e) => info!("No .env loaded ({e}); using process environment only"),
        }
        Self::from_env(std::env::vars())
    }

    /// Pure logic, testable without touching process env.
    fn from_env<I>(vars: I) -> Result<Config, String>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let map: HashMap<String, String> = vars.into_iter().collect();

        let deepseek_token = map
            .get("DEEPSEEK_TOKEN")
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                "DEEPSEEK_TOKEN is not set. Put it in .env or export it: \
                 DEEPSEEK_TOKEN=<userToken from chat.deepseek.com localStorage>"
                    .to_string()
            })?;

        let stt_model = map
            .get("LIHO_STT_MODEL")
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("models/t-one.onnx"));

        let system_prompt = map
            .get("LIHO_SYSTEM_PROMPT")
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

        Ok(Config {
            deepseek_token,
            stt_model,
            system_prompt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn loads_token_from_env() {
        let cfg = Config::from_env(vars(&[("DEEPSEEK_TOKEN", "abc123")])).unwrap();
        assert_eq!(cfg.deepseek_token, "abc123");
    }

    #[test]
    fn trims_whitespace_around_token() {
        let cfg = Config::from_env(vars(&[("DEEPSEEK_TOKEN", "  abc123 \n")])).unwrap();
        assert_eq!(cfg.deepseek_token, "abc123");
    }

    #[test]
    fn rejects_blank_token() {
        assert!(Config::from_env(vars(&[("DEEPSEEK_TOKEN", "   ")])).is_err());
    }

    #[test]
    fn missing_token_error_mentions_var_name() {
        let err = Config::from_env(vars(&[])).unwrap_err();
        assert!(err.contains("DEEPSEEK_TOKEN"));
    }

    #[test]
    fn stt_defaults_to_t_one() {
        let cfg = Config::from_env(vars(&[("DEEPSEEK_TOKEN", "t")])).unwrap();
        assert_eq!(cfg.stt_model, std::path::PathBuf::from("models/t-one.onnx"));
        assert!(cfg.system_prompt.contains("интервью"));
    }

    #[test]
    fn stt_model_overrides_from_env() {
        let cfg = Config::from_env(vars(&[
            ("DEEPSEEK_TOKEN", "t"),
            ("LIHO_STT_MODEL", "/opt/t-one.onnx"),
        ]))
        .unwrap();
        assert_eq!(cfg.stt_model, std::path::PathBuf::from("/opt/t-one.onnx"));
    }

    #[test]
    fn system_prompt_overrides_from_env() {
        let cfg = Config::from_env(vars(&[
            ("DEEPSEEK_TOKEN", "t"),
            ("LIHO_SYSTEM_PROMPT", "Отвечай на английском."),
        ]))
        .unwrap();
        assert_eq!(cfg.system_prompt, "Отвечай на английском.");
    }

    #[test]
    fn blank_prompt_falls_back_to_default() {
        let cfg = Config::from_env(vars(&[
            ("DEEPSEEK_TOKEN", "t"),
            ("LIHO_SYSTEM_PROMPT", "  "),
        ]))
        .unwrap();
        assert_eq!(cfg.system_prompt, DEFAULT_SYSTEM_PROMPT);
    }
}
