
export async function safeFetch<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);

    if (!response.ok) {
        const errorText = await response.text();
        // Check if the response is HTML (likely an error page)
        if (errorText.trim().startsWith('<!doctype html>') || errorText.trim().startsWith('<!DOCTYPE html>')) {
            throw new Error(`Received HTML response from backend: ${response.status} - ${errorText.substring(0, 200)}...`);
        }
        throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
        const responseText = await response.text();
        throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
    }

    return response.json();
}
