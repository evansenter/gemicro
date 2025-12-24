//! Integration tests for LlmClient against the real Gemini API
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! They are skipped if the API key is not set.

mod common;

use common::{create_test_client, get_api_key};
use futures_util::StreamExt;
use gemicro_core::LlmRequest;

#[tokio::test]
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
            println!("Response: {}", resp.text);
            println!("Tokens used: {:?}", resp.tokens_used);
            println!("Interaction ID: {}", resp.interaction_id);

            // Basic assertions
            assert!(!resp.text.is_empty(), "Response text should not be empty");
            assert!(
                resp.text.contains('4'),
                "Response should contain '4', got: {}",
                resp.text
            );
            assert!(
                !resp.interaction_id.is_empty(),
                "Interaction ID should not be empty"
            );
        }
        Err(e) => {
            panic!("Generate failed: {:?}", e);
        }
    }
}

#[tokio::test]
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
            println!("Response: {}", resp.text);

            assert!(!resp.text.is_empty(), "Response text should not be empty");
            // The response should mention Paris
            assert!(
                resp.text.to_lowercase().contains("paris"),
                "Response should mention Paris, got: {}",
                resp.text
            );
        }
        Err(e) => {
            panic!("Generate with system instruction failed: {:?}", e);
        }
    }
}

#[tokio::test]
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
            println!("Grounded response: {}", resp.text);
            println!("Tokens used: {:?}", resp.tokens_used);

            // Basic assertions - response should not be empty
            assert!(!resp.text.is_empty(), "Response text should not be empty");

            // The grounded response should contain date-related content
            // (we can't check for exact date since it may vary)
            assert!(
                resp.text.to_lowercase().contains("2024")
                    || resp.text.to_lowercase().contains("2025")
                    || resp.text.to_lowercase().contains("december")
                    || resp.text.to_lowercase().contains("january")
                    || resp.text.to_lowercase().contains("today"),
                "Grounded response should contain date-related content, got: {}",
                resp.text
            );
        }
        Err(e) => {
            panic!("Google Search grounding request failed: {:?}", e);
        }
    }
}
