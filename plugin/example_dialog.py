import os

import pcbnew
import numpy as np
from collections import defaultdict

import wx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from .version import __version__


class ExampleDialog(wx.Dialog):
    def __init__(self: "ExampleDialog", parent: wx.Frame) -> None:
        super().__init__(parent, -1, "KiCross")

        information_section = self.get_information_section()

        buttons = self.CreateButtonSizer(wx.OK)

        header = wx.BoxSizer(wx.HORIZONTAL)
        header.Add(information_section, 3, wx.ALL, 5)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(header, 0, wx.EXPAND | wx.ALL, 5)
        box.Add(buttons, 0, wx.EXPAND | wx.ALL, 5)
            
        self.SetSizerAndFit(box)



    def get_information_section(self) -> wx.BoxSizer:

        def calculate_via_counts(trk_start, trk_end, vias, radius=1.0):
            signal_count = 0
            ground_count = 0
            power_count = 0

            # Lista de pontos a verificar
            track_points = [trk_start, trk_end]

            for point in track_points:
                point_x, point_y = point.x, point.y

                for via in vias:
                    via_pos = via.GetPosition()/25400
                    via_x = via_pos.x
                    via_y = via_pos.y
                    via_netname = via.GetNetname()

                    # Calcular distância e verificar proximidade
                    dist_squared = (point_x - via_x) ** 2 + (point_y - via_y) ** 2
                    if dist_squared <= radius ** 2:  # Usando distância ao quadrado para evitar sqrt
                        if "gnd" in via_netname.lower():
                            ground_count += 1
                        elif any(keyword in via_netname.lower() for keyword in ["vcc", "power", "3v3", "+3v3"]):
                            power_count += 1
                        else:
                            signal_count += 1

            return signal_count, ground_count, power_count

        class RegressionModel(nn.Module):

            def __init__(self, input_dim):
                super().__init__()

                self.model = nn.Sequential(
                    nn.Linear(input_dim, 5),
                    # nn.BatchNorm1d(512),
                    # nn.Dropout(0.5),
                    nn.ReLU(),

                    nn.Linear(5, 32),
                    # nn.BatchNorm1d(128),
                    # nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Dropout(0.5),

                    nn.Linear(32, 32),
                    # nn.BatchNorm1d(128),
                    # nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Dropout(0.5),

                    nn.Linear(32, 32),
                    # nn.BatchNorm1d(128),
                    # nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Dropout(0.5),

                    nn.Linear(32, 1),
                    # # nn.BatchNorm1d(128),
                    # # nn.Dropout(0.5),
                    # nn.ReLU(),
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)
                output = self.model(x)
                return output
            
        class SimulationDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
                    
        pcb = pcbnew.GetBoard()

        nLayers = pcb.GetCopperLayerCount()

        source_dir = os.path.dirname(__file__)
        icon_file_name = os.path.join(source_dir, "icon.png")
        icon = wx.Image(icon_file_name, wx.BITMAP_TYPE_ANY)
        icon_bitmap = wx.Bitmap(icon)
        static_icon_bitmap = wx.StaticBitmap(self, wx.ID_ANY, icon_bitmap)

        font = wx.Font(
            12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
        )
        name = wx.StaticText(self, -1, "KiCad Crosstalk Analysis Plugin")
        name.SetFont(font)


        # ======================================================================================================================================

        pcb = pcbnew.GetBoard()
        
        nLayers = pcb.GetCopperLayerCount()
        thickness_diel = 630/(nLayers) # mils

        input_data = []
        label = []
        tracks = []

        c = 2540
        l = 1000

        vias = [v for v in pcb.GetTracks() if v.Type() == pcbnew.PCB_VIA_T]

        for item in pcb.Tracks():
            if item.Type() == pcbnew.PCB_TRACE_T:
                start_pos = (int(item.GetStart().x / c), int(item.GetStart().y)/c)
                end_pos = (int(item.GetEnd().x / c), int(item.GetEnd().y)/c)
                width = item.GetWidth()/ c
                net = item.GetNetname()
            
                tracks.append((start_pos, end_pos, width, net))

        def addVias(trails):

            trails_with_vias = []

            for trail in trails:
                if len(trail) > 1:
                                
                    for via in vias:
                        via_pos = (int(via.GetPosition().x / c), int(via.GetPosition().y / c))

                        if via_pos == trail:
                            trails_with_vias.append(trail)
                            break

            return trails_with_vias

        def build_tracks(segments):

            net_data = {}

            connectivity_graph = defaultdict(list)
            segment_properties = {} 

            for start, end, w, n in segments:

                connectivity_graph[start].append(end)
                connectivity_graph[end].append(start)
                segment_properties[(start, end)] = (w, n)
                segment_properties[(end, start)] = (w, n)

            def dfs(node, visited):

                trail = []
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        trail.append(current)
                        stack.extend(connectivity_graph[current])
                return trail

            visited = set()

            for node in connectivity_graph:
                if node not in visited:
                    trail = dfs(node, visited)

                    largura_set = set()
                    net_set = set()

                    for i in range(len(trail) - 1):
                        start, end = trail[i], trail[i + 1]
                        if (start, end) in segment_properties:
                            w, n = segment_properties[(start, end)]
                            largura_set.add(w)
                            net_set.add(n)

                    largura = largura_set.pop() if(len(largura_set) > 0) else None
                    net = net_set.pop() if(len(net_set) > 0) else None

                    comprimento = calculate_track_length(trail)
                    via = calculate_via_counts(trail, vias, radius=l)

                    if(largura and net):
                        net_data[net] = {
                            "length": comprimento,
                            "signal_amount": via[0],
                            "ground_amount": via[1],
                            "power_amount": via[2],
                            "width": largura,

                        }
                    
            return net_data

        def calculate_track_length(points):

            total_length = 0

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                total_length += segment_length

            return int(total_length)

        def calculate_via_counts(positions, vias, radius=1.0):
            signal_count = 0
            ground_count = 0
            power_count = 0

            for via in vias:
                via_pos = via.GetPosition()/c
                via_x = via_pos.x
                via_y = via_pos.y
                via_netname = via.GetNetname()

                for position in positions:

                    point_x, point_y = position[0], position[1]
                    dist = np.sqrt((point_x - via_x) ** 2 + (point_y - via_y) ** 2)

                    if dist <= radius and dist > 0:
                        if "gnd" in via_netname or "GND" in via_netname:
                            ground_count += 1
                        elif "vcc" in via_netname or "power" in via_netname or "VCC" in via_netname or "3V3" in via_netname or "+3V3" in via_netname:
                            power_count += 1
                        else:
                            signal_count += 1

            return signal_count, ground_count, power_count


        net_data  = build_tracks(tracks)


        for netname, data in net_data.items():
            
            track_params = [
                4.5,                # Permissividade
                1.0 / 1.68e-7,      # Condutividade
                thickness_diel,     # TDIEL
                1.38,               # TMET
                nLayers,            # Layer amount
                data["length"],     # Comprimento acumulado
                data["width"],      # Largura (última encontrada)
                data["signal_amount"],                  
                data["ground_amount"], 
                data["power_amount"]
            ]

            print(track_params)
            input_data.append(track_params)
            label.append(netname)

        path = os.path.join(r"C:\Users\otavi\OneDrive\Documentos\GitHub\KiCross\model", "model_weights.pth")

        model = RegressionModel(input_dim=10)
        model.load_state_dict(torch.load(path, weights_only = True))
        model.eval()


        class SimulationDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]


        unique_labels = list(set(label))
        label_to_index = {netname: idx for idx, netname in enumerate(unique_labels)}

        numeric_labels = [label_to_index[netname] for netname in label]

        X_test_tensor = torch.tensor(input_data, dtype=torch.float32)
        y_test_tensor = torch.tensor(numeric_labels, dtype=torch.float32).view(-1, 1)

        test_dataset = SimulationDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        predictions = []
        logits = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                logits.append(outputs)


        print(logits)
        logits = torch.cat(logits, dim=0)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        predictions = predictions.numpy()


        status = {0: "Passou", 1: "Falhou"}

        prints = []


        for idx, params in enumerate(input_data):

            value = int(predictions[idx])

            if(value == 1):
                prints.append(
                    f"Trilha - {label[idx]} - {status[value]} ",
                )

                 
        if len(prints) == 0:
            prints.append(f"Nenhuma trilha com potencial crosstalk")
        # # ======================================================================================================================================

        name_box = wx.BoxSizer(wx.HORIZONTAL)
        name_box.Add(static_icon_bitmap, 0, wx.ALL, 5)
        name_box.Add(name, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(name_box, 0, wx.ALL, 5)

        for v in prints:
            text = wx.StaticText(self, -1, v)
            box.Add(text, 0, wx.ALL, 5)

        return box
    
    
