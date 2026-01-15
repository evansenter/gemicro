//! Integration tests for LlmClient against the real Gemini API
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! They are skipped if the API key is not set.

mod common;

use common::{create_test_client, get_api_key, validate_response_semantically};
use futures_util::StreamExt;

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_simple_prompt() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let request = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_text("What is 2 + 2? Reply with just the number.")
        .build()
        .unwrap();

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            let tokens_used = resp.usage.as_ref().and_then(|u| u.total_tokens);
            println!("Response: {}", text);
            println!("Tokens used: {:?}", tokens_used);
            println!("Interaction ID: {:?}", resp.id);

            // Basic assertions
            assert!(!text.is_empty(), "Response text should not be empty");
            assert!(
                text.contains('4'),
                "Response should contain '4', got: {}",
                text
            );
            if let Some(id) = &resp.id {
                assert!(!id.is_empty(), "Interaction ID should not be empty");
            }
        }
        Err(e) => {
            panic!("Generate failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_with_system_instruction() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let request = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_system_instruction("You are a helpful assistant. Always respond in exactly one word.")
        .with_text("What is the capital of France?")
        .build()
        .unwrap();

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            println!("Response: {}", text);

            assert!(!text.is_empty(), "Response text should not be empty");
            // The response should mention Paris
            assert!(
                text.to_lowercase().contains("paris"),
                "Response should mention Paris, got: {}",
                text
            );
        }
        Err(e) => {
            panic!("Generate with system instruction failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_stream_simple_prompt() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let builder = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_text("Count from 1 to 5, one number per line.");

    let stream = client.generate_stream(builder);
    futures_util::pin_mut!(stream);

    let mut full_text = String::new();
    let mut chunk_count = 0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                print!("{}", chunk.text); // Print as we receive for visibility
                full_text.push_str(&chunk.text);
                chunk_count += 1;
            }
            Err(e) => {
                panic!("Stream error: {:?}", e);
            }
        }
    }
    println!(); // Newline after streaming output

    println!("Total chunks received: {}", chunk_count);
    println!("Full response: {}", full_text);

    // Assertions
    assert!(!full_text.is_empty(), "Streamed text should not be empty");
    assert!(chunk_count > 0, "Should have received at least one chunk");

    // Check that the response contains the expected numbers
    for num in 1..=5 {
        assert!(
            full_text.contains(&num.to_string()),
            "Response should contain {}, got: {}",
            num,
            full_text
        );
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_stream_with_system_instruction() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let builder = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_system_instruction("You are a pirate. Always respond in pirate speak.")
        .with_text("Say hello");

    let stream = client.generate_stream(builder);
    futures_util::pin_mut!(stream);

    let mut full_text = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                full_text.push_str(&chunk.text);
            }
            Err(e) => {
                panic!("Stream error: {:?}", e);
            }
        }
    }

    println!("Pirate response: {}", full_text);

    assert!(!full_text.is_empty(), "Streamed text should not be empty");
    // Pirate speak often includes these words
    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("ahoy")
            || lower.contains("matey")
            || lower.contains("arr")
            || lower.contains("ye")
            || lower.contains("captain")
            || lower.contains("hello"),
        "Response should contain pirate-like greeting, got: {}",
        full_text
    );
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_google_search_grounding() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);

    // Use a query that benefits from real-time web data
    let request = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_text("What is today's date?")
        .with_google_search()
        .build()
        .unwrap();

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            let tokens_used = resp.usage.as_ref().and_then(|u| u.total_tokens);
            println!("Grounded response: {}", text);
            println!("Tokens used: {:?}", tokens_used);

            // Basic assertions - response should not be empty
            assert!(!text.is_empty(), "Response text should not be empty");

            // The grounded response should contain date-related content
            // (we can't check for exact date since it may vary)
            assert!(
                text.to_lowercase().contains("2024")
                    || text.to_lowercase().contains("2025")
                    || text.to_lowercase().contains("2026")
                    || text.to_lowercase().contains("december")
                    || text.to_lowercase().contains("january")
                    || text.to_lowercase().contains("today"),
                "Grounded response should contain date-related content, got: {}",
                text
            );
        }
        Err(e) => {
            panic!("Google Search grounding request failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_structured_output_response_format() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);

    // Define a simple JSON schema for structured output
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level from 0 to 1"
            }
        },
        "required": ["answer", "confidence"]
    });

    let request = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .with_text("What is the capital of France? Respond with high confidence.")
        .with_response_format(schema)
        .build()
        .unwrap();

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            println!("Structured response: {}", text);

            // Parse the response as JSON
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(text);
            assert!(
                parsed.is_ok(),
                "Response should be valid JSON, got: {}",
                text
            );

            let json = parsed.unwrap();
            assert!(
                json.get("answer").is_some(),
                "Response should have 'answer' field"
            );
            assert!(
                json.get("confidence").is_some(),
                "Response should have 'confidence' field"
            );

            // Check the answer mentions Paris
            let answer = json["answer"].as_str().unwrap_or("");
            assert!(
                answer.to_lowercase().contains("paris"),
                "Answer should mention Paris, got: {}",
                answer
            );

            // Check confidence is a number
            assert!(
                json["confidence"].is_number(),
                "Confidence should be a number"
            );
        }
        Err(e) => {
            panic!("Structured output request failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_with_turns() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);

    // Use ConversationBuilder for clean multi-turn syntax
    let request = client
        .client()
        .interaction()
        .with_model("gemini-3-flash-preview")
        .conversation()
        .user("What is 2 + 2?")
        .model("2 + 2 equals 4.")
        .user("And what is that multiplied by 3? Just the number please.")
        .done()
        .build()
        .unwrap();

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            println!("Multi-turn response: {}", text);

            assert!(!text.is_empty(), "Response text should not be empty");

            // Use semantic validation instead of brittle string matching.
            // The model should understand "that" refers to 4, and return 12.
            // But it might phrase it as "12", "twelve", "The answer is 12", etc.
            let (is_valid, reason) = validate_response_semantically(
                &client,
                "User established that 2+2=4, then asked 'what is that multiplied by 3?'",
                text,
                "Does this response correctly indicate that the answer is 12 (4 * 3)?",
            )
            .await
            .expect("Semantic validation request failed");

            assert!(
                is_valid,
                "Response should correctly answer 4 * 3 = 12. Validation reason: {}. Response was: {}",
                reason,
                text
            );
        }
        Err(e) => {
            panic!("Multi-turn conversation failed: {:?}", e);
        }
    }
}
