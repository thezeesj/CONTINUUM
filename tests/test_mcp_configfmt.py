"""Tests for the config format layer (issue #1597).

``continuum mcp install`` writes into editors' own configuration files. The
failure this module exists to prevent is not a crash: it is a *quiet* rewrite
of a file this tool does not understand, where the user gets back something
that parses but is no longer what they wrote. So the tests below are mostly
refusal tests. Each one pins a file that must be reported and left alone.

The TOML and YAML tests skip when the optional parser is absent, because the
format layer deliberately does not depend on one: it refuses with an install
hint instead. A skip here means the parser is missing on this machine, not
that the format is unsupported.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from continuum.mcp import configfmt

#: A registration shaped the way the JSON hosts spell it. Every format has to
#: carry the same structure, because what a host reads is the structure and
#: not the bytes.
REGISTRATION = {
    "mcpServers": {
        "continuum-mcp": {
            "command": "/opt/venv/bin/continuum-mcp",
            "args": ["--db", "/home/op/proj/continuum.db"],
            "env": {"CONTINUUM_MCP_MUTATING_CLIENTS": "claude-code"},
        }
    }
}

needs_toml = pytest.mark.skipif(
    importlib.util.find_spec("tomli_w") is None,
    reason="writing TOML needs tomli-w; the format layer refuses without it by design",
)
needs_yaml = pytest.mark.skipif(
    importlib.util.find_spec("yaml") is None,
    reason="YAML support needs PyYAML; the format layer refuses without it by design",
)


def round_trips(fmt: str, data: dict[str, Any]) -> bool:
    """``data`` survives a write and a read in ``fmt`` unchanged."""
    rendered = configfmt.render_document(fmt, data)
    return configfmt._parse(fmt, rendered, Path("round-trip")) == data


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_json_round_trips_the_registration() -> None:
    """What a JSON host reads back is what was written, exactly."""
    assert round_trips("json", REGISTRATION)


@needs_yaml
def test_yaml_round_trips_the_registration() -> None:
    """Same promise for YAML, which has no JSON to lean on."""
    assert round_trips("yaml", REGISTRATION)


@needs_toml
def test_toml_round_trips_the_registration() -> None:
    """Same promise for TOML, the one format with no standard-library writer."""
    assert round_trips("toml", REGISTRATION)


@pytest.mark.parametrize("fmt", ["json", "toml", "yaml"])
def test_every_format_is_registered(fmt: str) -> None:
    """A profile naming a format this module does not know is a bug, not a crash.

    ``FORMATS`` is what a profile is validated against, so an unknown format
    has to be visible rather than falling through to a default.
    """
    assert fmt in configfmt.FORMATS


def test_an_unknown_format_is_refused_by_name() -> None:
    """A typo in a profile names itself instead of raising something opaque."""
    with pytest.raises(configfmt.ConfigError, match="unknown config format 'yml'"):
        configfmt.render_document("yml", {})


# --------------------------------------------------------------------------- #
# refusal: the file itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fmt", "broken", "expected"),
    [
        ("json", "{not json", "is not valid JSON"),
        ("json", "[]\n", "does not contain a JSON object"),
        ("json", '"just a string"\n', "does not contain a JSON object"),
    ],
)
def test_a_json_file_the_command_cannot_read_is_never_rewritten(
    tmp_path: Path, fmt: str, broken: str, expected: str
) -> None:
    """The refusal contract from ``_load_object``, preserved verbatim per format.

    A settings file holds unrelated configuration, so every shape that is not
    the one expected is reported rather than replaced. The wording is pinned
    because it is what the CLI prints to the operator.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(broken, encoding="utf-8")

    with pytest.raises(configfmt.ConfigError, match=expected):
        configfmt.read_document(fmt, settings)

    assert settings.read_text(encoding="utf-8") == broken


@needs_yaml
def test_a_yaml_file_that_will_not_parse_is_never_rewritten(tmp_path: Path) -> None:
    """Unparseable YAML is reported, not recreated.

    ``tab`` is the cheapest unparseable YAML there is: indentation may not
    contain a tab, so it fails in every YAML parser.
    """
    settings = tmp_path / "config.yaml"
    settings.write_text("mcpServers:\n\t- name: x\n", encoding="utf-8")

    with pytest.raises(configfmt.ConfigError, match="is not valid YAML"):
        configfmt.read_document("yaml", settings)

    assert settings.read_text(encoding="utf-8") == "mcpServers:\n\t- name: x\n"


@needs_yaml
def test_a_yaml_document_that_is_a_list_is_never_rewritten(tmp_path: Path) -> None:
    """A YAML file whose root is a list is refused the same way JSON's is."""
    settings = tmp_path / "config.yaml"
    settings.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(configfmt.ConfigError, match="does not contain a YAML mapping"):
        configfmt.read_document("yaml", settings)

    assert settings.read_text(encoding="utf-8") == "- one\n- two\n"


@needs_toml
def test_a_toml_file_that_will_not_parse_is_never_rewritten(tmp_path: Path) -> None:
    """Unparseable TOML is reported, not recreated."""
    settings = tmp_path / "config.toml"
    settings.write_text("[mcp_servers\n", encoding="utf-8")

    with pytest.raises(configfmt.ConfigError, match="is not valid TOML"):
        configfmt.read_document("toml", settings)

    assert settings.read_text(encoding="utf-8") == "[mcp_servers\n"


@needs_toml
def test_a_toml_array_value_is_read_as_a_table_not_refused(tmp_path: Path) -> None:
    """A key holding an array is a table, so it is read rather than refused.

    TOML has no root-level array: every document parses to a table, and
    ``key = [1, 2, 3]`` is the table ``{key: [1, 2, 3]}``. JSON and YAML both
    admit a non-mapping at the root, which is why they can be refused here;
    TOML cannot, so the ``does not contain a TOML table`` branch is
    unreachable for this format and pinning it would assert the opposite of
    the correct behaviour.
    """
    settings = tmp_path / "config.toml"
    settings.write_text("key = [1, 2, 3]\n", encoding="utf-8")

    assert configfmt.read_document("toml", settings) == {"key": [1, 2, 3]}
    assert settings.read_text(encoding="utf-8") == "key = [1, 2, 3]\n"


def test_an_absent_file_reads_as_an_empty_mapping(tmp_path: Path) -> None:
    """A config the editor has not written yet is not an error.

    This is what lets install create a registration in a fresh profile, and it
    is the one "empty" case that is folded rather than refused.
    """
    assert configfmt.read_document("json", tmp_path / "absent.json") == {}


@needs_yaml
def test_a_blank_yaml_file_reads_as_an_empty_mapping(tmp_path: Path) -> None:
    """A zero-byte YAML file parses to null; that is an editor, not a mistake."""
    settings = tmp_path / "config.yaml"
    settings.write_text("\n\n", encoding="utf-8")

    assert configfmt.read_document("yaml", settings) == {}


# --------------------------------------------------------------------------- #
# refusal: the value
# --------------------------------------------------------------------------- #


def test_toml_has_no_null_and_says_where_one_is(tmp_path: Path) -> None:
    """A null in a TOML document is refused by path, not dropped.

    Dropping the key would change what the host reads without saying so, and
    inventing a placeholder would change it differently. Both are worse than
    an error that names the key.
    """
    document = {"mcp_servers": {"continuum-mcp": {"command": "x", "cwd": None}}}

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.write_document("toml", tmp_path / "config.toml", document)

    assert "mcp_servers.continuum-mcp.cwd" in str(excinfo.value)
    assert "TOML has no null" in str(excinfo.value)
    assert not (tmp_path / "config.toml").exists(), "a refused write must leave no file"


# --------------------------------------------------------------------------- #
# refusal: what a rewrite would destroy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", ["toml", "yaml"])
@needs_yaml
def test_a_commented_file_is_refused_rather_than_flattened(tmp_path: Path, fmt: str) -> None:
    """Comments are the user's; a re-serialisation would delete them silently.

    Comments do not survive parsing at all in either format, so there is no
    way to write one back through a document model. That makes an existing
    commented file the one case where a whole-file write is destructive, and
    it is reported instead.
    """
    settings = tmp_path / f"config.{fmt}"
    marker = "# a note the operator wrote\n"
    settings.write_text(marker, encoding="utf-8")

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.write_document(fmt, settings, REGISTRATION)

    assert "comments" in str(excinfo.value)
    assert settings.read_text(encoding="utf-8") == marker


def test_json_does_not_refuse_on_comments_json_has_none(tmp_path: Path) -> None:
    """JSON has no comment syntax, so the JSON path never trips that guard.

    The guard is the reason every JSON host can still be rewritten wholesale,
    and a JSON string that merely *contains* a hash is not a comment.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"note": "colour #ff0000"}), encoding="utf-8")

    configfmt.write_document("json", settings, REGISTRATION)

    assert json.loads(settings.read_text(encoding="utf-8")) == REGISTRATION


def test_a_hash_inside_a_string_is_not_treated_as_a_comment(tmp_path: Path) -> None:
    """The comment guard looks at the first character of a line, nothing else.

    A smarter scan would try to tell a quoted ``#`` from a comment marker and
    would misfire on the very files it exists to protect. A line whose first
    non-blank character is ``#`` is the only thing counted.
    """
    assert not configfmt._has_comment_line('key: "value # not a comment"\n')
    assert configfmt._has_comment_line("  # indented comment\n")


# --------------------------------------------------------------------------- #
# refusal: an absent optional parser
# --------------------------------------------------------------------------- #


def test_a_missing_yaml_parser_is_reported_with_the_command_that_fixes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PyYAML means a named remedy, not a traceback.

    ``mcp install`` already refuses for a missing MCP SDK the same way, so the
    two failures read identically to an operator: here is what is missing,
    here is the command.
    """
    settings = tmp_path / "config.yaml"
    settings.write_text("mcpServers: []\n", encoding="utf-8")
    real_import_module = configfmt.importlib.import_module

    def without_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(configfmt.importlib, "import_module", without_yaml)

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.read_document("yaml", settings)

    message = str(excinfo.value)
    assert "PyYAML" in message
    assert 'pip install "continuum-agent[mcp]"' in message


def test_a_missing_toml_writer_is_reported_with_the_command_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for the TOML writer, which the standard library lacks."""
    real_import_module = configfmt.importlib.import_module

    def without_tomli_w(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tomli_w":
            raise ImportError("No module named 'tomli_w'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(configfmt.importlib, "import_module", without_tomli_w)

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.render_document("toml", REGISTRATION)

    message = str(excinfo.value)
    assert "tomli-w" in message
    assert 'pip install "continuum-agent[mcp]"' in message


def test_reading_toml_never_needs_a_writer() -> None:
    """``tomllib`` is standard library, so reading a TOML config always works.

    Only writing needs the extra. An operator who wants to inspect their file,
    or a ``mcp remove`` on a host this tool cannot write, must not be blocked
    by the missing writer.
    """
    import tomllib

    parsed = tomllib.loads('[mcp_servers.x]\ncommand = "c"\n')
    assert parsed == {"mcp_servers": {"x": {"command": "c"}}}


def test_reading_a_toml_file_goes_through_tomllib_itself(tmp_path: Path) -> None:
    """The lazy writer never enters a read: ``read_document`` reads TOML."""
    settings = tmp_path / "config.toml"
    settings.write_text('[mcp_servers.x]\ncommand = "c"\n', encoding="utf-8")

    assert configfmt.read_document("toml", settings) == {"mcp_servers": {"x": {"command": "c"}}}


def test_an_unsupported_format_is_refused_by_name(tmp_path: Path) -> None:
    """A format the layer does not know is named, next to the ones it does."""
    settings = tmp_path / "config.ini"
    settings.write_text("[mcp_servers]\n", encoding="utf-8")

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.read_document("ini", settings)

    assert str(excinfo.value) == "unknown config format 'ini' (expected one of json, toml, yaml)"


@pytest.mark.parametrize(
    ("fmt", "text", "label"),
    [
        pytest.param("json", "[1, 2, 3]", "a JSON object", id="json-list"),
        pytest.param("json", '"a string"', "a JSON object", id="json-scalar"),
        pytest.param("yaml", "- one\n- two\n", "a YAML mapping", id="yaml-list"),
        pytest.param("yaml", "42", "a YAML mapping", id="yaml-scalar"),
    ],
)
def test_a_document_that_is_not_a_mapping_is_refused(
    fmt: str, text: str, label: str, tmp_path: Path
) -> None:
    """A structured file this tool cannot model is reported, never replaced.

    TOML is absent from the params on purpose: ``tomllib`` always produces a
    table at the root, so the non-mapping branch cannot be reached through it.
    The check itself is shared by every format, so JSON and YAML pin it.
    """
    settings = tmp_path / f"config.{fmt}"
    settings.write_text(text, encoding="utf-8")

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.read_document(fmt, settings)

    assert str(excinfo.value) == f"{settings} does not contain {label}; refusing to edit it"


@needs_yaml
def test_a_yaml_document_holding_only_an_explicit_null_is_refused(tmp_path: Path) -> None:
    """``---`` parses to None but is not blank, so it is not folded to empty."""
    settings = tmp_path / "config.yaml"
    settings.write_text("---\n", encoding="utf-8")

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.read_document("yaml", settings)

    assert str(excinfo.value) == f"{settings} is empty; refusing to edit it"


@needs_yaml
def test_an_uncommented_yaml_file_is_still_rewritable(tmp_path: Path) -> None:
    """The comment guard checks the file; it does not assume one is there.

    A clean existing file must still be writable, or the guard would have
    become a blanket refusal on every YAML host with a settings file. The
    write replaces the whole document, so the read-merge-write flow an
    install actually takes is what has to keep the operator's own keys.
    """
    settings = tmp_path / "config.yaml"
    settings.write_text("other: 1\n", encoding="utf-8")

    data = configfmt.read_document("yaml", settings)
    LIST_SPLIT.put(data, "local", Path("/proj"), SPEC)
    configfmt.write_document("yaml", settings, data)

    reparsed = configfmt.read_document("yaml", settings)
    assert reparsed["other"] == 1
    assert "mcpServers" in reparsed


def test_an_uncommented_toml_file_is_still_rewritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guard on the TOML side, through the writer the import resolves."""
    settings = tmp_path / "config.toml"
    settings.write_text("other = 1\n", encoding="utf-8")

    writer = _RecordingWriter()
    with _tomli_w_resolved_to(writer, monkeypatch):
        data = configfmt.read_document("toml", settings)
        DICT_SPLIT.put(data, "local", Path("/proj"), SPEC)
        configfmt.write_document("toml", settings, data)

    assert settings.read_text(encoding="utf-8") == _RENDERED_BY_THE_WRITER
    assert writer.seen == [
        {
            "other": 1,
            "projects": {str(Path("/proj")): {"mcpServers": {SPEC.name: DICT_SPLIT.entry(SPEC)}}},
        }
    ]


@pytest.mark.parametrize(
    ("data", "null_at"),
    [
        pytest.param({"a": [None]}, "a[0]", id="null-in-a-list"),
        pytest.param({"a": [{"b": [None]}]}, "a[0].b[0]", id="null-deeper-in-a-list"),
    ],
)
def test_a_null_inside_a_list_is_named_by_its_dotted_path(
    data: dict[str, Any], null_at: str
) -> None:
    """TOML has no null, and one hiding inside a list is refused by path."""
    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.render_document("toml", data)

    assert str(excinfo.value) == (
        f"{null_at} is null, and TOML has no null to write it as; "
        "removing the key first would change what the host reads"
    )


#: What the recording writer below answers every ``dumps`` call with. A fixed
#: string the test can compare against: ``render_document`` returns what the
#: writer produced, verbatim.
_RENDERED_BY_THE_WRITER = "rendered-by-the-writer\n"


class _RecordingWriter:
    """Stands in for the optional ``tomli-w`` package.

    Tomli-w is an install-time extra this layer deliberately does not depend
    on, so a machine without it cannot exercise the dump itself. The module
    the lazy import resolves to is all ``render_document`` ever asks of the
    package, so substituting one records what it was handed and what it said.
    """

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def dumps(self, data: dict[str, Any]) -> str:
        self.seen.append(data)
        return _RENDERED_BY_THE_WRITER


@contextlib.contextmanager
def _tomli_w_resolved_to(
    writer: _RecordingWriter, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_RecordingWriter]:
    """Resolve ``import_module("tomli_w")`` to ``writer`` for the block."""
    real_import_module = configfmt.importlib.import_module

    def with_the_writer(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tomli_w":
            return writer
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(configfmt.importlib, "import_module", with_the_writer)
    yield writer


def test_render_toml_delegates_the_dump_to_the_resolved_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dump is the imported module's, and its result is returned as-is."""
    writer = _RecordingWriter()

    with _tomli_w_resolved_to(writer, monkeypatch):
        rendered = configfmt.render_document("toml", REGISTRATION)

    assert rendered == _RENDERED_BY_THE_WRITER
    assert writer.seen == [REGISTRATION]


def test_a_missing_yaml_writer_is_reported_with_the_command_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing YAML needs the same package reading it does, and says so.

    The sibling test pins the read path; this one drives ``render_document``,
    which is what an install into a Continue-shaped host actually takes.
    """
    real_import_module = configfmt.importlib.import_module

    def without_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(configfmt.importlib, "import_module", without_yaml)

    with pytest.raises(configfmt.ConfigError) as excinfo:
        configfmt.render_document("yaml", REGISTRATION)

    assert str(excinfo.value) == (
        "reading and writing YAML needs the 'PyYAML' package, which is not "
        'installed. Install it with: pip install "continuum-agent[mcp]"'
    )


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #

#: The shapes the landed profiles use. Every combination here is real: no
#: profile is dict-keyed with an argv array, but a YAML host being list-shaped
#: and a TOML host being dict-keyed both are, so the axes are exercised
#: independently rather than only in the combinations that happen to ship.
DICT_SPLIT = configfmt.Shape(
    servers_key="mcpServers", nesting_key="projects", type_key="type", type_value="stdio"
)
LIST_SPLIT = configfmt.Shape(servers_key="mcpServers", container="list")
DICT_ARRAY = configfmt.Shape(
    servers_key="mcp",
    container="dict",
    argv_style="array",
    env_key="environment",
    type_key="type",
    type_value="local",
)

SPEC = configfmt.ServerSpec(
    name="continuum-mcp",
    argv=("/opt/venv/bin/continuum-mcp", "--db", "/home/op/proj/continuum.db"),
    env={"CONTINUUM_MCP_MUTATING_CLIENTS": "claude-code"},
)


@pytest.mark.parametrize("shape", [DICT_SPLIT, LIST_SPLIT, DICT_ARRAY])
def test_a_registration_survives_a_write_and_a_read_in_any_shape(shape: configfmt.Shape) -> None:
    """Every shape reads back the same registration install wrote.

    This is the round-trip guarantee for the structure layer, and it is the
    reason the shape does not depend on the format: the argv that goes in is
    the argv that comes back, whatever the host's keys are called.
    """
    data: dict[str, Any] = {}
    assert shape.put(data, "local", Path("/proj"), SPEC)[0] == "installed"

    assert shape.argv_of(shape.find(data, "local", Path("/proj"), SPEC.name)) == list(SPEC.argv)


@pytest.mark.parametrize("shape", [DICT_SPLIT, LIST_SPLIT, DICT_ARRAY])
def test_re_installing_never_duplicates_the_entry(shape: configfmt.Shape) -> None:
    """The second install is a no-op, on every shape.

    This is the bug class of issues #484 and #526. A dict shape makes it
    invisible, because a key either exists or it does not; a list shape
    appends, so getting it wrong shows up as two identical servers in the
    agent's picker and nothing else anywhere.
    """
    data: dict[str, Any] = {}
    assert shape.put(data, "project", Path("/proj"), SPEC)[0] == "installed"
    assert shape.put(data, "project", Path("/proj"), SPEC)[0] == "present"

    container = shape.container_at(data, "project", Path("/proj"), create=False)[1]
    entries = [container] if isinstance(container, dict) else container
    assert len(entries) == 1


@pytest.mark.parametrize("shape", [DICT_SPLIT, LIST_SPLIT, DICT_ARRAY])
def test_a_moved_environment_repoints_rather_than_duplicating(shape: configfmt.Shape) -> None:
    """A changed argv updates in place; it never leaves a second entry.

    Reinstalling after a virtualenv moves has to be safe to run
    unconditionally from a setup script, and a list-shaped host that appends
    on every run is exactly how that becomes a problem.
    """
    moved = configfmt.ServerSpec(
        name=SPEC.name,
        argv=("/new/venv/bin/continuum-mcp", "--db", SPEC.argv[2]),
        env=SPEC.env,
    )
    data: dict[str, Any] = {}
    shape.put(data, "project", Path("/proj"), SPEC)

    assert shape.put(data, "project", Path("/proj"), moved)[0] == "updated"

    container = shape.container_at(data, "project", Path("/proj"), create=False)[1]
    entries = [container] if isinstance(container, dict) else container
    assert len(entries) == 1
    assert shape.argv_of(shape.find(data, "project", Path("/proj"), SPEC.name)) == list(moved.argv)


@pytest.mark.parametrize("shape", [DICT_SPLIT, LIST_SPLIT, DICT_ARRAY])
def test_remove_takes_out_only_what_was_installed(shape: configfmt.Shape) -> None:
    """Other servers, and other keys, survive a remove untouched."""
    data: dict[str, Any] = {"unrelated": {"keep": True}}
    shape.put(data, "project", Path("/proj"), SPEC)
    container = shape.container_at(data, "project", Path("/proj"), create=False)[1]
    if shape.container == "list":
        container.append({"name": "weather", "command": "weather-bin"})
    else:
        container["weather"] = {"command": "weather-bin"}

    assert shape.drop(data, "project", Path("/proj"), SPEC.name) is True

    assert shape.find(data, "project", Path("/proj"), SPEC.name) is None
    assert data["unrelated"] == {"keep": True}
    assert shape.find(data, "project", Path("/proj"), "weather") is not None


def test_a_nested_registration_prunes_every_container_it_emptied() -> None:
    """A local registration sits three containers deep; all three go.

    The alternative is a per-user settings file left holding an empty project
    keyed by an absolute path, which is residue a user notices and files a bug
    about.
    """
    data: dict[str, Any] = {}
    DICT_SPLIT.put(data, "local", Path("/proj"), SPEC)
    assert data == {
        # The container is keyed by str(project_root), which renders per
        # platform: "/proj" on POSIX, "\proj" on Windows. Asserting the POSIX
        # spelling made this fail on Windows while the code was right.
        "projects": {str(Path("/proj")): {"mcpServers": {"continuum-mcp": DICT_SPLIT.entry(SPEC)}}}
    }

    DICT_SPLIT.drop(data, "local", Path("/proj"), SPEC.name)

    assert data == {}


def test_an_empty_list_container_is_left_alone() -> None:
    """``mcpServers: []`` survives, because the host may have written it.

    Deleting the key would remove the only thing telling Continue that MCP is
    configured at all.
    """
    data: dict[str, Any] = {"mcpServers": []}
    LIST_SPLIT.put(data, "local", Path("/proj"), SPEC)

    LIST_SPLIT.drop(data, "local", Path("/proj"), SPEC.name)

    assert data == {"mcpServers": []}


@pytest.mark.parametrize(
    ("taken", "shape", "expected"),
    [
        pytest.param(["not a dict"], DICT_SPLIT, "is not an object", id="dict-taken-by-a-list"),
        pytest.param({"not": "a list"}, LIST_SPLIT, "is not a list", id="list-taken-by-a-mapping"),
    ],
)
def test_a_container_that_is_the_wrong_type_is_refused(
    taken: Any, shape: configfmt.Shape, expected: str
) -> None:
    """A container taken by something else is never overwritten.

    The same contract ``_load_object`` has for the file itself, one level
    down: whatever is in there was written on purpose.
    """
    data = {shape.servers_key: taken}

    with pytest.raises(configfmt.ConfigError, match=expected):
        shape.put(data, "project", Path("/proj"), SPEC)


def test_a_nesting_container_that_is_the_wrong_type_is_refused() -> None:
    """A ``projects`` key holding a string stops the install dead."""
    data: dict[str, Any] = {"projects": "a string"}

    with pytest.raises(configfmt.ConfigError, match="is not an object"):
        DICT_SPLIT.put(data, "local", Path("/proj"), SPEC)


def test_a_shape_without_per_project_nesting_ignores_the_scope() -> None:
    """A host with nowhere to nest puts every scope in the same container.

    opencode and Continue scope a registration by which file it is in, not by
    a section inside one, so ``local``, ``project`` and ``user`` are the same
    place and the scope only picked the path in the first place.
    """
    data: dict[str, Any] = {}
    for scope in ("local", "project", "user"):
        LIST_SPLIT.put(data, scope, Path("/proj"), SPEC)

    assert list(data) == ["mcpServers"]
    assert len(data["mcpServers"]) == 1


def test_find_returns_none_where_the_container_itself_is_absent() -> None:
    """Nothing is filed under a name when the container does not exist yet.

    For the nested shape the missing container is a nesting level deep; for
    the flat one it is the servers key. Both must answer None rather than
    raising, so a remove on a host that was never configured is a no-op.
    """
    assert DICT_SPLIT.find({}, "local", Path("/proj"), SPEC.name) is None
    assert LIST_SPLIT.find({}, "local", Path("/proj"), SPEC.name) is None


def test_drop_on_a_name_that_was_never_installed_changes_nothing() -> None:
    """A removal finds nothing, reports False, and creates nothing doing it."""
    data: dict[str, Any] = {}
    assert DICT_SPLIT.drop(data, "local", Path("/proj"), SPEC.name) is False
    # The search did not build the nesting levels it walked looking for it.
    assert data == {}


def test_prune_skips_a_level_that_already_emptied_itself() -> None:
    """A key missing from its parent is a level already gone, not a stop.

    The reverse walk may meet a level a preceding deletion has already
    removed; that level is skipped and the one above it is the next
    candidate, so a prune never stops short of the residue it exists to take.
    """
    data: dict[str, Any] = {"a": {"b": {}}}
    configfmt._prune(data, ["a", "x"])
    assert data == {"a": {"b": {}}}


def test_argv_of_rejects_every_entry_shape_it_cannot_represent() -> None:
    """``None`` means "not argv", whether that is a type error or a bad value.

    The caller cannot act differently on a missing ``args`` and on an ``args``
    holding a number, so both have to produce the same refusal: guessing here
    is what makes a remove delete somebody else's configuration.
    """
    assert DICT_SPLIT.argv_of("not a dict") is None
    assert DICT_SPLIT.argv_of({"command": "/x/continuum-mcp"}) is None
    assert DICT_SPLIT.argv_of({"command": "/x/continuum-mcp", "args": [1]}) is None
    assert DICT_SPLIT.argv_of({"command": 7, "args": []}) is None
    assert DICT_ARRAY.argv_of({"command": "/x/continuum-mcp"}) is None
    assert DICT_ARRAY.argv_of({"command": []}) is None
    assert DICT_ARRAY.argv_of({"command": ["/x/continuum-mcp", 2]}) is None


# --------------------------------------------------------------------------- #
# host profiles
# --------------------------------------------------------------------------- #

from continuum.mcp import install as mcp_install  # noqa: E402

#: What each host's real config file looks like on arrival, copied from the
#: vendor's own documentation or a shipped fixture. A profile that reads the
#: wrong key, or writes the wrong one, fails against these even when every
#: shape test above passes, which is the point of keeping them: the shape
#: tests prove the layer works, the fixtures prove the profile is right.
HOST_FIXTURES: dict[str, dict[str, Any]] = {
    # https://zed.dev/docs/ai/mcp
    "zed": {
        "context_servers": {
            "local-mcp-server": {"command": "some-command", "args": ["a", "b"], "env": {}}
        }
    },
    # https://opencode.ai/docs/mcp-servers/
    "opencode": {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {"jira": {"type": "remote", "url": "https://jira/mcp"}},
    },
    # https://github.com/continuedev/continue/blob/main/docs/customize/deep-dives/mcp.mdx
    "continue": {
        "name": "My assistant",
        "version": "0.0.1",
        "schema": "v1",
        "mcpServers": [
            {"name": "SQLite MCP", "type": "stdio", "command": "npx", "args": ["mcp-sqlite", "/db"]}
        ],
    },
    # https://developers.openai.com/codex/config-reference
    "codex": {
        "model": "gpt-5",
        "mcp_servers": {"github": {"command": "npx", "args": ["-y", "gh-mcp"]}},
    },
}

#: Formats that need an optional parser, so a run without it reports the
#: profile as unsupported instead of failing on an ImportError.
_FORMAT_GUARD = {
    "toml": importlib.util.find_spec("tomli_w") is not None,
    "yaml": importlib.util.find_spec("yaml") is not None,
}


def profile_runnable(host: str) -> bool:
    """Whether this machine can write ``host``'s format at all."""
    return _FORMAT_GUARD.get(mcp_install.host_format(host), True)


PROFILES = sorted(mcp_install.HOST_PROFILES)


@pytest.mark.parametrize("host", PROFILES)
def test_every_profile_declares_the_keys_the_layers_read(host: str) -> None:
    """A profile missing a key fails here, not as a KeyError mid-write.

    The shape and format lookups are the first thing every host path does, so
    an incomplete row would otherwise only surface when somebody ran
    ``mcp install --host`` against it.
    """
    profile = mcp_install.HOST_PROFILES[host]
    for key in (
        "project_settings",
        "local_settings",
        "user_settings",
        "mutating_clients",
        "format",
        "container",
        "servers_key",
        "nested_by_project",
        "argv_style",
        "env_key",
        "type_key",
        "type_value",
    ):
        assert key in profile, (host, key)
    assert profile["format"] in configfmt.FORMATS, host
    assert profile["container"] in ("dict", "list"), host
    assert profile["argv_style"] in ("split", "array"), host


@pytest.mark.parametrize("host", PROFILES)
def test_an_install_lands_where_the_vendor_says_it_lands(host: str) -> None:
    """The written file has the container the vendor's own example shows.

    For the four JSON hosts this runs everywhere. TOML and YAML hosts are
    exercised through their fixture and shape directly below, because their
    parsers are optional and a missing one is a refusal, not a defect.
    """
    if not profile_runnable(host):
        pytest.skip(f"{host} needs an optional parser for {mcp_install.host_format(host)}")
    if mcp_install.host_format(host) != "json":
        pytest.skip(f"{host} writes {mcp_install.host_format(host)}, covered by the fixture tests")

    shape = mcp_install.host_shape(host)
    fixture = HOST_FIXTURES.get(host)
    if fixture is None:
        # The four hosts that shipped before this layer have no fixture here:
        # their config files are covered end to end in test_mcp_install.py,
        # against the real command the vendor documents.
        data: dict[str, Any] = {}
    else:
        # A private copy: the fixtures are module-level, and every test below
        # writes into the document it is handed.
        data = json.loads(json.dumps(fixture))

    path = shape.path("project", Path("/proj"))
    if fixture is not None:
        assert data[path[-1]] is not None, (host, path, sorted(data))

    spec = configfmt.ServerSpec("continuum-mcp", ("/bin/x", "--db", "/p/d.db"), {"K": "v"})
    entry = shape.entry(spec)
    shape.put(data, "project", Path("/proj"), spec)

    assert shape.argv_of(shape.find(data, "project", Path("/proj"), "continuum-mcp")) == [
        "/bin/x",
        "--db",
        "/p/d.db",
    ]
    assert entry


@pytest.mark.parametrize(
    ("host", "fixture"),
    [
        ("zed", HOST_FIXTURES["zed"]),
        ("opencode", HOST_FIXTURES["opencode"]),
        ("continue", HOST_FIXTURES["continue"]),
        ("codex", HOST_FIXTURES["codex"]),
    ],
)
def test_the_other_servers_in_a_real_config_survive_an_install(
    host: str, fixture: dict[str, Any]
) -> None:
    """Installing beside a host's own servers leaves all of them alone.

    The fixture is that host's documented config with its documented server
    already in it. Ours goes in beside it and a remove takes only ours back
    out, which is the property a user with a real config depends on and the
    one a shape test against an empty document cannot show.
    """
    shape = mcp_install.host_shape(host)
    data = json.loads(json.dumps(fixture))  # a private copy per parametrisation

    status, previous = shape.put(
        data,
        "project",
        Path("/proj"),
        configfmt.ServerSpec("continuum-mcp", ("/bin/x", "--db", "/p/d.db"), {"K": "v"}),
    )

    assert status == "installed" and previous is None
    assert shape.drop(data, "project", Path("/proj"), "continuum-mcp") is True
    assert data == fixture, (host, data)


@pytest.mark.parametrize("host", PROFILES)
def test_a_foreign_entry_under_our_name_is_never_overwritten(host: str) -> None:
    """Recognition is per-host, and it has to be narrow on every host.

    ``mcp remove`` deletes on this predicate, so a shape that accepted
    somebody else's entry would delete it. A list-shaped host is the sharpest
    case: an entry it does not own is a record in a list, not a key, and
    getting that wrong either drops a stranger's server or leaves ours.
    """
    foreign = {
        "command": "/opt/manual/continuum-mcp",
        "args": ["--db", "continuum.db"],
        "env": {"CONTINUUM_MCP_MUTATING_CLIENTS": "someone-else"},
    }
    assert mcp_install._is_managed_server(foreign, host) is False


def test_the_verified_client_names_are_the_ones_the_sources_send() -> None:
    """The two allowlists that were read out of a host's source, pinned.

    Zed sending "Zed" with a capital Z is the whole argument for this check
    existing: the obvious guess from the product name is wrong, and a wrong
    one costs the host ten of its thirteen tools with nothing else changing.
    opencode's is the case where the obvious guess happens to be right, which
    is exactly why guessing is not a method.
    """
    assert mcp_install.HOST_PROFILES["zed"]["mutating_clients"] == "Zed"
    assert mcp_install.HOST_PROFILES["opencode"]["mutating_clients"] == "opencode"
