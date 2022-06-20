"""
Example request to interact with the webservice

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

import requests

data = {"data": [
    {'doi-XXX':"Globally, the number of people living in extreme poverty declined from 36 per cent in 1990 to 10 per cent in 2015. " },
    {'doi-YYY': "Education enables upward socioeconomic mobility and is a key to escaping poverty. Over the past decade, major progress was made towards increasing access to education and school enrollment rates at all levels, particularly for girls. "} ]}


res = requests.post('http://localhost:6007/sdg', json=data)
print(res.json())

