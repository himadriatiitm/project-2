from dataclasses import dataclass
from utils import truncate_string
from typing import Any, List

@dataclass
class IOBlock:
    code: str
    result: Any


@dataclass
class IOChain:
    chain: List[IOBlock]
    limit: int = 8

    def __repr__(self) -> str:
        repr = ""
        for block in self.chain[-self.limit :]:
            code = truncate_string(block.code, 80)
            output_annotation = "output"
            if isinstance(block.result, Exception):
                output_annotation = "error"
            result = truncate_string(block.result, 80)

            repr += f"""
```python
{code}
```

```{output_annotation}
{result}
```
"""
        return repr

    def dump(self) -> str:
        return "\n".join(block.code for block in self.chain[:-1])

    def dump_with(self, new_code: str) -> str:
        return "\n".join((self.dump(), new_code))

    def last_is_err(self) -> bool:
        return isinstance(self.chain[-1].result, Exception)

