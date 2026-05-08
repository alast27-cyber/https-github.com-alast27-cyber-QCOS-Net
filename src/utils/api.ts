
export async function safeFetch<T>(url: string, options?: RequestInit, retries = 10, backoff = 1500): Promise<T> {
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

    const contentType = response.headers.get('content-type');
    const responseText = await response.text();

    if (!response.ok) {
        // Check if the response is HTML and contains "Starting Server" or is a common platform standby page
        const isStartupHTML = responseText.toLowerCase().includes('starting server') || 
                            responseText.toLowerCase().includes('please wait') ||
                            responseText.toLowerCase().includes('application building');

        if (isStartupHTML && retries > 0) {
            console.warn(`[API] Server is starting or building, retrying in ${backoff}ms... (${retries} retries left)`);
            await new Promise(resolve => setTimeout(resolve, backoff));
            return safeFetch(url, options, retries - 1, backoff * 1.5);
        }

        console.error(`[API] Fetch error for ${url}: ${response.status} - ${responseText.substring(0, 200)}...`);
        throw new Error(`API error: ${response.status} - ${responseText.substring(0, 200)}...`);
    }

    if (!contentType || !contentType.includes('application/json')) {
        // Check if the response is HTML and contains "Starting Server" or similar
        const isStartupHTML = responseText.toLowerCase().includes('starting server') || 
                            responseText.toLowerCase().includes('please wait') ||
                            responseText.toLowerCase().includes('application building') ||
                            responseText.trim().startsWith('<!doctype html>');

        if (isStartupHTML && retries > 0) {
            console.warn(`[API] Server returned HTML (likely standby/boot), retrying in ${backoff}ms... (${retries} retries left)`);
            await new Promise(resolve => setTimeout(resolve, backoff));
            return safeFetch(url, options, retries - 1, backoff * 1.5);
        }

        console.error(`[API] Expected JSON for ${url} but got ${contentType}: ${responseText.substring(0, 200)}...`);
        throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
    }

    try {
        return JSON.parse(responseText);
    } catch (e) {
        throw new Error(`Failed to parse JSON response for ${url}. Body segment: ${responseText.substring(0, 200)}...`);
    }
}
