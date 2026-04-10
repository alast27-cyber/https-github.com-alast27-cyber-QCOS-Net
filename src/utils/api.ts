
export async function safeFetch<T>(url: string, options?: RequestInit, retries = 5, backoff = 1000): Promise<T> {
    console.log(`[API] Fetching ${url}`);
    let response;
    try {
        response = await fetch(url, options);
    } catch (e) {
        console.error(`[API] Fetch failed for ${url}:`, e);
        if (retries > 0) {
            console.warn(`Connection failed, retrying in ${backoff}ms...`);
            await new Promise(resolve => setTimeout(resolve, backoff));
            return safeFetch(url, options, retries - 1, backoff * 2);
        }
        throw e;
    }

    if (response.status === 429 && retries > 0) {
        console.warn(`Rate limited, retrying in ${backoff}ms...`);
        await new Promise(resolve => setTimeout(resolve, backoff));
        return safeFetch(url, options, retries - 1, backoff * 2);
    }

    if (!response.ok) {
        const errorText = await response.text();
        console.error(`[API] Fetch error for ${url}: ${response.status} - ${errorText}`);
        // Check if the response is HTML (likely an error page or "Starting Server...")
        if (errorText.trim().toLowerCase().startsWith('<!doctype html>')) {
            if (errorText.includes('Starting Server...')) {
                if (retries > 0) {
                    console.warn(`Server starting, retrying in ${backoff}ms...`);
                    await new Promise(resolve => setTimeout(resolve, backoff));
                    return safeFetch(url, options, retries - 1, backoff * 2);
                }
            }
            throw new Error(`Received HTML response from backend: ${response.status} - ${errorText.substring(0, 200)}...`);
        }
        throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
        const responseText = await response.text();
        console.error(`[API] Expected JSON for ${url} but got ${contentType}: ${responseText.substring(0, 200)}...`);
        // Check if the response is HTML (likely "Starting Server...")
        if (responseText.trim().toLowerCase().includes('starting server...')) {
            if (retries > 0) {
                console.warn(`Server starting, retrying in ${backoff}ms...`);
                await new Promise(resolve => setTimeout(resolve, backoff));
                return safeFetch(url, options, retries - 1, backoff * 2);
            }
        }
        throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
    }

    return response.json();
}
