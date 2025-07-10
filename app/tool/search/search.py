from __future__ import annotations
from typing import Callable, Sequence, Any, Optional
from app.tool.search.base import WebSearchEngine, SearchItem
from pydantic import ConfigDict, Field, model_validator

class GenericWebSearchEngine(WebSearchEngine):
    name: str
    description: str = Field(
        default=None,
        description="A human‑readable description of this engine."
    )
    search_fn: Callable[[str, int], Sequence[Any]] = Field(
        ...,
        description="The function to call for performing the search."
    )
    title_field: str = Field("title", description="Which key in result items holds the title")
    url_field: str = Field("url", description="Which key in result items holds the URL")
    desc_field: str = Field("description", description="Which key in result items holds the snippet")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _populate_and_validate(cls, model: GenericWebSearchEngine) -> GenericWebSearchEngine:
        if model.description is None:
            model.description = f"Generic engine for {model.name}"
            
        for fld in ("title_field", "url_field", "desc_field"):
            v = getattr(model, fld)
            if not isinstance(v, str) or not v:
                raise ValueError(f"{fld!r} must be a non‐empty string")
        return model

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> list[SearchItem]:
        raw_results = self.search_fn(query, num_results)
        items: list[SearchItem] = []

        for idx, r in enumerate(raw_results):
            title = url = desc = None

            if isinstance(r, str):
                url   = r
                title = f"{self.name} Result {idx+1}"
            elif isinstance(r, dict):
                title = r.get(self.title_field, f"{self.name} Result {idx+1}")
                url   = r.get(self.url_field,   "")
                desc  = r.get(self.desc_field,  None)
            else:
                title = getattr(r, self.title_field, f"{self.name} Result {idx+1}")
                url   = getattr(r, self.url_field,   "")
                desc  = getattr(r, self.desc_field,  None)

            if not url:
                url = str(r)

            items.append(SearchItem(title=title, url=url, description=desc))

        return items[:num_results]
