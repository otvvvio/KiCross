import os

import pcbnew
import wx

from .example_dialog import ExampleDialog


class TemplatePluginAction(pcbnew.ActionPlugin):
    def defaults(self) -> None:
        self.name = "KiCross"
        self.category = "Analysis"
        self.description = "This plugin run a AI based crosstalk analysis"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")

    def Run(self) -> None:
        pcb_frame = next(
            x for x in wx.GetTopLevelWindows() if x.GetName() == "PcbFrame"
        )

        dlg = ExampleDialog(pcb_frame)
        if dlg.ShowModal() == wx.ID_OK:

            pass

        dlg.Destroy()
