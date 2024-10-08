# flake8: noqa
NEWS_DOCS = """API documentation:
Endpoint: https://newsapi.org
Everything /v2/everything
Search through millions of articles from over 150,000 large and small news sources and blogs.

This endpoint suits article discovery and analysis.

#####################################################

Request parameters:

    q | string | Keywords or phrases to search for in the article title and body.
    Advanced search is supported here:
    Surround phrases with quotes (") for exact match.
    Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
    Prepend words that must not appear with a - symbol. Eg: -bitcoin
    Alternatively you can use the AND / OR / NOT keywords, and optionally group these with parenthesis. Eg: crypto AND (ethereum OR litecoin) NOT bitcoin.
    The complete value for q must be URL-encoded. Max length: 500 chars.

    pageSize | int | The number of results to return per page (request). 20 is the default, 100 is the maximum.
    page | int | Use this to page through the results if the total results found is greater than the page size.

######################################################

Response object:
    status | string | If the request was successful or not. Options: ok, error. In the case of error a code and message property will be populated.
    totalResults | int | The total number of results available for your request.
    articles | array[article] | The results of the request.
    source | object | The identifier id and a display name name for the source this article came from.
    author | string | The author of the article
    title | string | The headline or title of the article.
    description | string | A description or snippet from the article.
    url | string | The direct URL to the article.
    urlToImage | string | The URL to a relevant image for the article.
    publishedAt | string | The date and time that the article was published, in UTC (+000)
    content | string | The unformatted content of the article, where available. This is truncated to 200 chars.
"""
