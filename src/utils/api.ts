
export async function safeFetch<T>(url: string, options?: RequestInit, retries = 3, backoff = 1000): Promise<T> {
    const response = await fetch(url, options);

    if (response.status === 429 && retries > 0) {
        console.warn(`Rate limited, retrying in ${backoff}ms...`);
        await new Promise(resolve => setTimeout(resolve, backoff));
        return safeFetch(url, options, retries - 1, backoff * 2);
    }

    if (!response.ok) {
        const errorText = await response.text();
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
