//! Integration tests for LlmClient against the real Gemini API
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! They are skipped if the API key is not set.

mod common;

use common::{create_test_client, get_api_key};
use futures_util::StreamExt;
use gemicro_core::LlmRequest;

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_simple_prompt() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let request = LlmRequest::new("What is 2 + 2? Reply with just the number.");

    let response = client.generate(request).await;

    match response {
        Ok(resp) => {
            let text = resp.text().unwrap_or("");
            let tokens_used = resp.usage.as_ref().and_then(|u| u.total_tokens);
            println!("Response: {}", text);
            println!("Tokens used: {:?}", tokens_used);
            println!("Interaction ID: {}", resp.id);

            // Basic assertions
            assert!(!text.is_empty(), "Response text should not be empty");
            assert!(
                text.contains('4'),
                "Response should contain '4', got: {}",
                text
            );
            assert!(!resp.id.is_empty(), "Interaction ID should not be empty");
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
    let request = LlmRequest::with_system(
        "What is the capital of France?",
        "You are a helpful assistant. Always respond in exactly one word.",
    );

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
    let request = LlmRequest::new("Count from 1 to 5, one number per line.");

    let stream = client.generate_stream(request);
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
    let request = LlmRequest::with_system(
        "Say hello",
        "You are a pirate. Always respond in pirate speak.",
    );

    let stream = client.generate_stream(request);
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
async fn test_generate_empty_prompt_error() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let request = LlmRequest::new("");

    let response = client.generate(request).await;

    assert!(response.is_err(), "Empty prompt should return an error");

    if let Err(e) = response {
        println!("Expected error: {:?}", e);
        assert!(
            matches!(e, gemicro_core::LlmError::InvalidRequest(_)),
            "Should be InvalidRequest error, got: {:?}",
            e
        );
    }
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_generate_stream_empty_prompt_error() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let request = LlmRequest::new("");

    let stream = client.generate_stream(request);
    futures_util::pin_mut!(stream);

    // The first item from the stream should be an error
    let first = stream.next().await;

    assert!(first.is_some(), "Stream should yield at least one item");

    if let Some(Err(e)) = first {
        println!("Expected error: {:?}", e);
        assert!(
            matches!(e, gemicro_core::LlmError::InvalidRequest(_)),
            "Should be InvalidRequest error, got: {:?}",
            e
        );
    } else {
        panic!("Expected error for empty prompt");
    }
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
    let request = LlmRequest::new("What is today's date?").with_google_search();

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

    let request = LlmRequest::new("What is the capital of France? Respond with high confidence.")
        .with_response_format(schema);

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
