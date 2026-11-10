RETRYABLE_ERROR_CODES = frozenset(
    {
        "json_invalid",
        "missing",
        "union_tag_invalid",
        "union_tag_not_found",
    }
)


def is_retryable_error(error_code: str) -> bool:
    return error_code in RETRYABLE_ERROR_CODES
