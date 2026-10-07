from src.chunker import chunk_python, chunk_text


def test_chunk_python():
    content = """def hello():
    print("hello")


def goodbye():
    print("goodbye")
"""

    chunks = chunk_python(
        content,
        "test.py",
        2000,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert len(chunk.text) <= 2000
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_chunk_python_too_large():
    content = "def hello():\n" + "    print('hello')\n" * 100

    chunks = chunk_python(
        content,
        "test.py",
        100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_chunk_text():
    content = """First paragraph.

Second paragraph.

Third paragraph.
"""

    chunks = chunk_text(
        content,
        "test.md",
        100,
    )

    assert len(chunks) == 1

    for chunk in chunks:
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_invalid_python():
    content = "def hello(:"

    chunks = chunk_python(
        content,
        "test.py",
        2000,
    )

    assert chunks == []