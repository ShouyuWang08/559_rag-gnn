from __future__ import annotations

from dataclasses import dataclass

EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class Metapath:
    name: str
    edges: tuple[EdgeType, ...]

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def node_types(self) -> tuple[str, ...]:
        out = [self.edges[0][0]]
        for _, _, dst in self.edges:
            out.append(dst)
        return tuple(out)


# Four meta-paths whose edge types are confirmed present in Hetionet v1.0.
# Co-regulation paths (CuGuD / CdGdD / CuGdD / CdGuD) would require reverse
# traversal of Disease-upregulates/downregulates-Gene edges, which are stored
# as forward-only in the Hetionet JSON; they are therefore omitted here and
# left as future work (see §7).  CbGpPpG is also omitted because the Pathway
# hub does not connect back to Disease in the standard Hetionet metagraph.
METAPATHS: list[Metapath] = [
    Metapath("CpD", (
        ("Compound", "palliates", "Disease"),
    )),
    Metapath("CbGaD", (
        ("Compound", "binds", "Gene"),
        ("Gene", "associates", "Disease"),
    )),
    Metapath("CrCtD", (
        ("Compound", "resembles", "Compound"),
        ("Compound", "treats", "Disease"),
    )),
    Metapath("CbGiGaD", (
        ("Compound", "binds", "Gene"),
        ("Gene", "interacts", "Gene"),
        ("Gene", "associates", "Disease"),
    )),
]
