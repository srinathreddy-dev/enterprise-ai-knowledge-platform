import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const apiUrl = process.env.API_BASE_URL;
  const apiKey = process.env.APP_API_KEY;

  if (!apiUrl || !apiKey) {
    return NextResponse.json(
      { detail: "API configuration is missing on the server." },
      { status: 500 }
    );
  }

  let body: { question?: string };

  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON request." },
      { status: 400 }
    );
  }

  const question = body.question?.trim();

  if (!question || question.length > 2000) {
    return NextResponse.json(
      { detail: "Question must contain between 1 and 2000 characters." },
      { status: 422 }
    );
  }

  try {
    const upstream = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({ question }),
      cache: "no-store",
      signal: AbortSignal.timeout(45000),
    });

    const responseBody = await upstream.json();

    const response = NextResponse.json(responseBody, {
      status: upstream.status,
    });

    const retryAfter = upstream.headers.get("Retry-After");

    if (retryAfter) {
      response.headers.set("Retry-After", retryAfter);
    }

    return response;
  } catch (error) {
    console.error("Backend API request failed:", error);

    return NextResponse.json(
      { detail: "The AI service is temporarily unavailable." },
      { status: 502 }
    );
  }
}
