from __future__ import annotations

from types import SimpleNamespace

import torch

from icalens._capture import capture_resid_post


class AddOne(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
        return (values + 1,)


class HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([AddOne(), AddOne()])
        self.transformer.ln_f = torch.nn.Identity()

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = input_ids.float().unsqueeze(-1)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        final = hidden * 100  # Deliberately distinguish final-norm output.
        return SimpleNamespace(last_hidden_state=final)


def test_last_resid_post_is_captured_before_final_norm() -> None:
    model = HookModel()
    captured = capture_resid_post(
        model,
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        layers=(0, 1),
        positions=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(captured[0].squeeze(-1), torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(captured[1].squeeze(-1), torch.tensor([3.0, 4.0]))
