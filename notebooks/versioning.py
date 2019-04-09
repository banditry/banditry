import inspect


def hash_class(klass):
    """Hash the source code of the given class, ignoring comment lines.
    
    Notes:
        This implementation does not inspect super classes, so changes
        in a super class will not change the hash but may substantively
        change the behavior of the class.
    """
    source_lines = (l for l in inspect.getsource(klass).split('\n')
                    if l.strip())
    non_comment_source = ''.join(filter_out_comments(source_lines))
    return hash(non_comment_source)


def filter_out_comments(lines):
    """Filter out lines of source code which are comments.

    Args:
        lines (iterable[str]): lines of source code to filter.

    Returns:
        generator[str]: subset of `lines` which are not comments.
    """
    return (line for line, is_comment in annotate_comments(lines)
            if not is_comment)


def annotate_comments(lines):
    """Annotate each line of code with True if it's a comment or part of a comment, else False.

    Args:
        lines (iterable[str]): lines of source code to annotate.

    Returns:
        generator[tuple]: pairs of (original_line, is_comment_flag).
    """
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if in_multiline_comment:
            if stripped_line.endswith('"""'):
                in_multiline_comment = False
            yield line, True
        elif stripped_line.startswith('"""'):
            if stripped_line == '"""' or not stripped_line.endswith('"""'):
                in_multiline_comment = True
            yield line, True
        elif stripped_line.startswith('#'):
            yield line, True
        else:
            yield line, False
