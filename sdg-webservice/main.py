"""
SDG Webservice

Authors:
* Nick Jelicic (Dialogic)
* Tommy van der Vorst (Dialogic)
* Wilfred Mijnhardt (Rotterdam School of Management)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

import os
from flask import Flask, Response
from flask import request, make_response
from sdg_model import SDGModel, SDGDataset
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from transformers import BertConfig
import json

app = Flask(__name__)

model_config = BertConfig.from_pretrained('bert-base-uncased')
model_config.output_hidden_states = True
model = SDGModel(conf=model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('../models/model_2.bin', map_location=device)['model_state_dict'])

model.eval()
model.to(device)

def wrapped(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        resp = None
        try:
            with model_lock:
                resp = make_response(f(*args, **kwargs))

        except Exception as e:
            traceback.print_exc()
            resp = Response(json.dumps({"error": "exception %s" % e}))
            resp.status_code = 411
        # Exit on exception (so Docker can restart us)
        # os._exit(-1)

        resp.headers["Content-type"] = "application/json"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    return decorated_function


@app.route('/sdg', methods=['POST'])
def sdgModel():


    text = request.args.get('text')
    print(text)
    outs = []

    test_dataset = SDGDataset(
        abstract=text)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=16
    )
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            preds = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
            )

            outs.append(np.round(torch.sigmoid(preds).cpu().detach().numpy(), 2))

    outs = np.vstack(outs)
    output_dic = dict()
    for i in range(1, 18):
        key = 'SDG ' + str(i)
        value = str(outs[:, i - 1][0])
        output_dic[key] = value
    resp = Response(json.dumps(output_dic))
    return resp


if __name__ == "__main__":
    print("SDG server is running")
    app.run(host="0.0.0.0", port=5000)
