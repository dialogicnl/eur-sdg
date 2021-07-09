"""
Script that demonstrates how to use the functions in sdg_util.py

Authors:
* Nick Jelicic (Dialogic)
* Tommy van der Vorst (Dialogic)
* Bijan Ranjbar (MyDataExpert)
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

import warnings
warnings.filterwarnings('ignore')

from tqdm.autonotebook import tqdm
from sdg_util import *
import pandas as pd
import torch



#LOAD MODEL
tokenizer = tokenizers.BertWordPieceTokenizer(
    '../models/bert-base-uncased-vocab.txt', 
    lowercase=True
)

model_config = BertConfig.from_pretrained('bert-base-uncased')
model_config.output_hidden_states = True
model = SDGModel(conf=model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('../models/model_2.bin', map_location=device)['model_state_dict'])

model.eval()
model.to(device)


#SAMPLE DATA
sample_data = {
    'ID_1': """

Bottom trawling is a method of fishing that involves dragging heavy weighted nets across the sea floor, in an effort to catch fish. It’s a favoured method by commercial fishing companies because it can catch large quantities of product in one go.

In New Zealand, commercial fishing companies bottom trawl both within our waters and in international ones. In fact, we are one of only seven countries still using this practise in international waters. 
Bottom trawling by NZ fisheries
What are bottom trawlers fishing for?

New Zealand fishing companies are primarily bottom trawling for orange roughy and oreo.  These are deepwater species, often found around seamounts – submarine mountains. Most of the fish trawled by the New Zealand fleet are then exported to China or the United States. 
Why is bottom trawling destructive?

The problem with bottom trawling as a fishing method is that it’s indiscriminate in what it catches. When dragging the large, weighted nets across the seafloor, everything that happens to be in the way gets swept up in the net too. For this reason bottom trawling has a large bycatch impact, with many non target species being fished in the process. 

This has an impact on the biodiversity of the ocean, and also means many species are being fished to the brink simply as a consequence of commercial activities, not as the target of them. 

In addition to the turtles, juvenile fish and invertebrates that get swept up in trawling nets, deep sea corals are hidden victims of trawling.  

Deep sea coral forests, thought to be some of the most biodiverse ecosystems with high degree of endemism (species found only there), can take centuries to form. But when a trawler runs over them again and again to catch fish, they’re destroyed, and so is the whole community that had formed around them. 

Consider a forest on land. The ancient trees are worth saving alone, but when you bulldoze a forest you also lose the whole ecosystem of smaller plants and animals that have found a home there. We’ve all shed a tear watching the video of a solo orangutan, clinging to the last tree as bulldozers level its forest home. But this is happening in our oceans too.  What happens there is so out of sight, it’s simply out of mind for many.
Black seabream, yellow amberkacks, Mount Vema seamount, Seamount biodiversity, Greenpeace, Greenpeace pole to pole, Bottom trawling on seamounts
Shoals of black seabream and yellowtail amberjacks circle above one of Mount Vema’s summits.
Seamounts are known to be unique ecosystems.

Coral forests, coined the ‘kauri of our ocean,’ act as nurseries for juvenile fish and other invertebrates, and are often found on seamounts. These places are vital not just for the communities of fish, starfish, crabs, sea urchins, brittle stars, mollusks, sponges, and worms that live there, but are also thought to be vital stop off points for migratory species like whales. It’s thought some species of whales even use seamounts for navigation and to stop and feed during their long journeys. 

Trawlers target seamounts because they are known hot spots for fish and other marine life. The key species they are after often hang out on seamounts, so they trawl over them to get their catch.
Paragorgia coral dredged, bottom trawling coral, bottom trawling New Zealand, New Zealand bottom trawling vessel, ancient coral forests
Crewman on the New Zealand bottom trawler, Waipori, dump a large piece of Paragorgia coral dredged from the deep sea in their net.
Does bottom trawling still happen in New Zealand?

A common misconception is that New Zealand has banned bottom trawling. We haven’t. The industry volunteered to ‘protect’ a third of New Zealand waters, but most of the areas they chose were unsuitable for bottom trawling anyway. It was essentially protecting what couldn’t be touched even if they had wanted to. Meanwhile, many seamounts and hotspots for marine life are still trawled to this day – and last year the quota for Orange Roughy was increased, meaning even more trawling. 

When it comes to bottom trawling, New Zealand is one of the bad guys. We’re one of only seven countries that still bottom trawl in the High Seas – in those areas beyond national borders where laws are hazy. 

The UN’s advice (Resolution 61/105 – 64/72 – 66/68)  is that all vulnerable areas, with precious ecosystems, like sea mounts should be closed to bottom trawling. New Zealand signed on to these resolutions, but we are failing to live up to international expectations. 

New Zealand owned fishing companies, such as Sanford and Talley’s, trawl internationally and in New Zealand waters. A whole host of companies trawl within our EEZ.
Amaltal Apollo, Talley's, Talley's vessels, Talley's fisheries, bottom trawling boats, SPRFMO
The Amaltal Apollo, that was caught trawling in a protected area, pictured here in the Port of Nelson. © Jason Blair

On the domestic front, trawling has been blamed for the degradation of some of our coastal waters – and is at least partly to blame for the sorry state of the Hauraki Gulf.

Internationally, we don’t do much better. At the most recent meeting of SPRFMO (South Pacific Regional Fisheries Management Organisation) meeting, New Zealand lobbied for weaker regulations on trawling. Representatives for New Zealand lobbied for more coral bycatch to be allowed per trawl, while other nations tried hard to make the laws stricter to protect the oceans.

At this same meeting, New Zealand delegates lobbied to have a Talley’s owned fishing vessel, the Amaltal Apollo, removed from the fishing blacklist, after it was caught bottom trawling in a protected area. They won this appeal. 
Bottom trawling and the biodiversity crisis

At a time when the world faces an unprecedented biodiversity crisis, methods of fishing that cause damage to such slow growing ecosystems can no longer be afforded. Last year alone, the New Zealand fleet destroyed up to 3,000 tonnes of deep sea corals in their relentless trawling. 

Scientists tell us that we must protect a third of the world’s oceans if we want to save them from collapse, and that means protecting them from devastating activities like this.

If we want to prevent more species from going extinct, or joining the growing IUCN red list we must take decisive action to restrict activities that are known to destroy and disturb vital ecosystems that support life on Earth.
Reasons for hope 

For decades, Greenpeace has been campaigning hard to protect the oceans. 

Last month, a new study published in Nature revealed that ocean recovery is possible within the next three decades – so long as we start increasing protection now.

It states: “We are at a point at which we can choose between a legacy of a resilient and vibrant ocean or an irreversibly disrupted ocean, for the generations to follow.”

The study puts in no uncertain terms that we have the opportunity to protect our oceans for the future well-being of the planet, but it is going to take co-operation on an international scale.

A strong Global Ocean Treaty has the power to make ocean recovery possible over the next thirty years. Getting this across the line could pave the way for a network of ocean sanctuaries around the world, with the ability to protect unique ecosystems so they can recover.

You can be part of this vital movement by signing the petition here.

Greenpeace New Zealand is also running a campaign with our allies DSCC, WWF, Forest & Bird and ECO, to ban bottom trawling on seamounts – because these ecosystems are so vulnerable and slow growing. It’s our belief that these ancient coral forests should be protected from bottom trawling to enable them to continue supporting a diverse range of marine life.

Please sign the petition here. 

To remain independent, Greenpeace accepts no funding from government or business.
We rely on you. Make a donation now.
""" ,
    'ID_2': """ 
en.wikipedia.org
Smart city
A smart city is an urban area that uses different types of electronic methods and sensors to collect data. Insights gained from that data are used to manage assets, resources and services efficiently; in return, that data is used to improve the operations across the city. This includes data collected from citizens, devices, buildings and assets that is then processed and analyzed to monitor and manage traffic and transportation systems, power plants, utilities, water supply networks, waste, crime detection,[1] information systems, schools, libraries, hospitals, and other community services.[2][3]

The smart city concept integrates information and communication technology (ICT), and various physical devices connected to the IoT (Internet of things) network to optimize the efficiency of city operations and services and connect to citizens.[4][5] Smart city technology allows city officials to interact directly with both community and city infrastructure and to monitor what is happening in the city and how the city is evolving. ICT is used to enhance quality, performance and interactivity of urban services, to reduce costs and resource consumption and to increase contact between citizens and government.[6] Smart city applications are developed to manage urban flows and allow for real-time responses.[7] A smart city may therefore be more prepared to respond to challenges than one with a simple "transactional" relationship with its citizens.[8][9] Yet, the term itself remains unclear to its specifics and therefore, open to many interpretations.[10]
Terminology[edit]

Due to the breadth of technologies that have been implemented under the smart city label, it is difficult to distill a precise definition of a smart city. Deakin and Al Waer[11] list four factors that contribute to the definition of a smart city:

    The application of a wide range of electronic and digital technologies to communities and cities.
    The use of ICT to transform life and working environments within the region.
    The embedding of such Information and Communications Technologies (ICTs) in government systems.
    The territorialisation of practices that brings ICTs and people together to enhance the innovation and knowledge that they offer.

Deakin defines the smart city as one that utilises ICT to meet the demands of the market (the citizens of the city), and that community involvement in the process is necessary for a smart city.[12] A smart city would thus be a city that not only possesses ICT technology in particular areas, but has also implemented this technology in a manner that positively impacts the local community.

Alternative definitions include:

    Giffinger et al. 2007: "Regional competitiveness, transport and Information and Communication Technologies economics, natural resources, human and social capital, quality of life, and participation of citizens in the governance of cities."[13]
    Smart Cities Council[when?]: "A smart city is one that has digital technology embedded across all city functions."[14][full citation needed]
    Caragliu and Nijkamp 2009: "A city can be defined as 'smart' when investments in human and social capital and traditional (transport) and modern (ICT) communication infrastructure fuel sustainable economic development and a high quality of life, with a wise management of natural resources, through participatory action and engagement."[15]
    Frost & Sullivan 2014: "We identified eight key aspects that define a smart city: smart governance, smart energy, smart building, smart mobility, smart infrastructure, smart technology, smart healthcare and smart citizen."[16]
    Institute of Electrical and Electronics Engineers smart Cities: "A smart city brings together technology, government and society to enable the following characteristics: smart cities, a smart economy, smart mobility, a smart environment, smart people, smart living, smart governance."[17][when?]
    Business Dictionary: "A developed urban area that creates sustainable economic development and high quality of life by excelling in multiple key areas; economy, mobility, environment, people, living, and government. Excelling in these key areas can be done so through strong human capital, social capital, and/or ICT infrastructure."[18][when?]
    Indian Government 2014: "Smart city offers sustainability in terms of economic activities and employment opportunities to a wide section of its residents, regardless of their level of education, skills or income levels."[19]
    Department for Business, Innovation and Skills, UK 2013: "The concept is not static, there is no absolute definition of a smart city, no end point, but rather a process, or series of steps, by which cities become more 'liveable' and resilient and, hence, able to respond more quickly to new challenges."[20]

Characteristics[edit]

It has been suggested that a smart city (also community, business cluster, urban agglomeration or region) uses information technologies to:

    Make more efficient use of physical infrastructure (roads, built environment and other physical assets) through artificial intelligence and data analytics in order to support a strong and healthy economic, social, cultural development.[21]
    Engage effectively with local governance officials by use of open innovation processes and e-participation, improving the collective intelligence of the city's institutions through e-governance,[7] with emphasis placed on citizen participation and co-design.[22][23]
    Learn, adapt and innovate and thereby respond more effectively and promptly to changing circumstances by improving the intelligence of the city.[7][24]

They evolve towards a strong integration of all dimensions of human intelligence, collective intelligence, and also artificial intelligence within the city.[25][26] The intelligence of cities "resides in the increasingly effective combination of digital telecommunication networks (the nerves), ubiquitously embedded intelligence (the brains), sensors and tags (the sensory organs), and software (the knowledge and cognitive competence)".[27]

These forms of intelligence in smart cities have been demonstrated in three ways:

Bletchley Park often considered to be the first smart community.

    Orchestration intelligence:[7] Where cities establish institutions and community-based problem solving and collaborations, such as in Bletchley Park, where the Nazi Enigma cypher was decoded by a team led by Alan Turing. This has been referred to as the first example of a smart city or an intelligent community.[28]
    Empowerment intelligence: Cities provide open platforms, experimental facilities and smart city infrastructure in order to cluster innovation in certain districts. These are seen in the Kista Science City in Stockholm and the Cyberport Zone in Hong Kong. Similar facilities have also been established in Melbourne.[29]

    Hong Kong Cyberport 1 and Cyberport 2 Buildings
    A hub has been created in Kyiv, which develops public projects.
    Instrumentation intelligence: Where city infrastructure is made smart through real-time data collection, with analysis and predictive modelling across city districts. There is much controversy surrounding this, particularly with regards to surveillance issues in smart cities. Examples of Instrumentation intelligence have been implemented in Amsterdam.[30] This is implemented through:[7]
        A common IP infrastructure that is open to researchers to develop applications.
        Wireless meters and devices transmit information at the point in time.
        A number of homes being provided with smart energy meters to become aware of energy consumption and reduce energy usage.
        Solar power garbage compactors, car recharging stations and energy saving lamps.

Some major fields of intelligent city activation are:
Innovation economy 	Urban infrastructure 	Governance
Innovation in industries, clusters, districts of a city 	Transport 	Administration services to the citizen
Knowledge workforce: Education and employment 	Energy / Utilities 	Participatory and direct democracy
Creation of knowledge-intensive companies 	Protection of the environment / Safety 	Services to the citizen: Quality of life

According to David K. Owens, the former executive vice president of the Edison Electric Institute, two key elements that a smart city must have are an integrated communications platform and a "dynamic resilient grid." Both are large investments.[31]
Frameworks[edit]

The creation, integration, and adoption of smart city capabilities require a unique set of frameworks to realize the focus areas of opportunity and innovation central to smart city projects. The frameworks can be divided into 5 main dimensions which include numerous related categories of smart city development:[32]
Technology framework[edit]

A smart city relies heavily on the deployment of technology. Different combinations of technological infrastructure interact to form the array of smart city technologies with varying levels of interaction between human and technological systems.[33]

    Digital: A service oriented infrastructure is required to connect individuals and devices in a smart city. These include innovation services and communication infrastructure. Yovanof, G. S. & Hazapis, G. N. define a digital city as "a connected community that combines broadband communications infrastructure; a flexible, service-oriented computing infrastructure based on open industry standards; and, innovative services to meet the needs of governments and their employees, citizens and businesses."[34]
    Intelligent: Cognitive technologies, such as artificial intelligence and machine learning, can be trained on the data generated by connected city devices to identify patterns. The efficacy and impact of particular policy decisions can be quantified by cognitive systems studying the continuous interactions of humans with their urban surroundings.[4]
    Ubiquitous: A ubiquitous city provides access to public services through any connected device. U-city is an extension of the digital city concept because of the facility in terms of accessibility to every infrastructure.[35]
    Wired: The physical components of IT systems are crucial to early-stage smart city development. Wired infrastructure is required to support the IoT and wireless technologies central to more interconnected living.[36] A wired city environment provides general access to continually updated digital and physical infrastructure. The latest in telecommunications, robotics, IoT, and various connected technologies can then be deployed to support human capital and productivity.[37][38]
    Hybrid: A hybrid city is the combination of a physical conurbation and a virtual city related to the physical space. This relationship can be one of virtual design or the presence of a critical mass of virtual community participants in a physical urban space. Hybrid spaces can serve to actualize future-state projects for smart city services and integration.[39]
    Information city: The multiplicity of interactive devices in a smart city generates a large quantity of data. How that information is interpreted and stored is critical to Smart city growth and security.[40]

Human framework[edit]

Smart city initiatives have measurable positive impacts on the quality of life of its citizens and visitors.[41] The human framework of a smart city – its economy, knowledge networks, and human support systems – is an important indicator of its success.[42]

    Creativity: Arts and culture initiatives are common focus areas in smart city planning.[43][44] Innovation is associated with intellectual curiosity and creativeness, and various projects have demonstrated that knowledge workers participate in a diverse mix of cultural and artistic activities.[45][46]
    Learning: Since mobility is a key area of Smart city development, building a capable workforce through education initiatives is necessary.[42] A city's learning capacity includes its education system, including available workforce training and support, and its cultural development and exchange.[47]
    Humanity: Numerous Smart city programs focus on soft infrastructure development, like increasing access to voluntary organizations and designated safe zones.[48] This focus on social and relational capital means diversity, inclusion, and ubiquitous access to public services is worked in to city planning.[38]
    Knowledge: The development of a knowledge economy is central to Smart city projects.[49] Smart cities seeking to be hubs of economic activity in emerging tech and service sectors stress the value of innovation in city development.[38]

Institutional framework[edit]

According to Moser, M. A.,[47] since the 1990s, the smart communities movement took shape as a strategy to broaden the base of users involved in IT. Members of these Communities are people that share their interest and work in a partnership with government and other institutional organizations to push the use of IT to improve the quality of daily life as a consequence of different worsening in daily actions. Eger, J. M.[50] said that a smart community makes a conscious and agreed-upon decision to deploy technology as a catalyst to solving its social and business needs. It is very important to understand that this use of IT and the consequent improvement could be more demanding without the institutional help; indeed institutional involvement is essential to the success of smart community initiatives. Again Moser, M. A.[47] explained that "building and planning a smart community seeks for smart growth"; smart growth is essential for the partnership between citizen and institutional organizations to react to worsening trends in daily issues like traffic congestion, school overcrowding and air pollution. However, it is important to note that technological propagation is not an end in itself, but only a means to reinventing cities for a new economy and society. To sum up, it is possible to assert that any smart city initiatives necessitate the government's support for their success.

The importance of these three different dimensions is that only a link among them can make possible the development of a real smart city concept. According to the definition of smart city given by Caragliu, A., Del Bo, C., & Nijkamp, P.,[51] a city is smart when investments in human/social capital and IT infrastructure fuel sustainable growth and enhance quality of life, through participatory governance.
Energy framework[edit]

Smart cities use data and technology to create efficiencies, improve sustainability, create economic development, and enhance quality of life factors for people living and working in the city. It also means that the city has a smarter energy infrastructure. More formally, a smart city is: "… An urban area that has securely integrated technology across the information … and Internet of Things (IoT) sectors to better manage a city’s assets."[52] Employment of smart technologies enables the more efficient application of integrated energy technologies in the city allowing the development of more self-sustaining areas or even Positive Energy Districts that produce more energy than consume.[53]

A smart city is powered by "smart connections" for various items such as street lighting, smart buildings, distributed energy resources (DER), data analytics, and smart transportation. Amongst these things, energy is paramount; this is why utility companies play a key role in smart cities. Electric companies, working partnership with city officials, technology companies and a number of other institutions, are among the major players that helped accelerate the growth of America's smart cities.[54]
Data Management framework[edit]

Smart cities employ a combination of data collection, processing, and disseminating technologies in conjunction with networking and computing technologies and data security and privacy measures encouraging the application of innovation to promote the overall quality of life for its citizens and covering dimensions that include: utilities, health, transportation, entertainment and government services.[55]
Roadmap[edit]

A smart city roadmap consists of four/three (the first is a preliminary check) major components:[3][56]

    Define exactly what is the community: maybe that definition can condition what you are doing in the subsequent steps; it relates to geography, links between cities and countryside and flows of people between them; maybe – even – that in some Countries the definition of City/community that is stated does not correspond effectively to what – in fact – happens in real life.
    Study the Community: Before deciding to build a smart city, first we need to know why. This can be done by determining the benefits of such an initiative. Study the community to know the citizens, the business's needs – know the citizens and the community's unique attributes, such as the age of the citizens, their education, hobbies, and attractions of the city.
    Develop a smart city Policy: Develop a policy to drive the initiatives, where roles, responsibilities, objective, and goals, can be defined. Create plans and strategies on how the goals will be achieved.
    Engage The Citizens: This can be done by engaging the citizens through the use of e-government initiatives, open data, sport events, etc.

In short, People, Processes, and Technology (PPT) are the three principles of the success of a smart city initiative. Cities must study their citizens and communities, know the processes, business drivers, create policies, and objectives to meet the citizens' needs. Then, technology can be implemented to meet the citizens' need, in order to improve the quality of life and create real economic opportunities. This requires a holistic customized approach that accounts for city cultures, long-term city planning, and local regulations.

    "Whether to improve security, resiliency, sustainability, traffic congestion, public safety, or city services, each community may have different reasons for wanting to be smart. But all smart communities share common attributes—and they all are powered by smart connections and by our industry's smarter energy infrastructure. A smart grid is the foundational piece in building a smart community." – Pat Vincent-Collawn, chairman of the Edison Electric Institute and president and CEO of PNM Resources.[57]

Policies[edit]

ASEAN Smart Cities Network (ASCN) is a collaborative platform which aims to synergise Smart city development efforts across ASEAN by facilitating cooperation on smart city development, catalysing bankable projects with the private sector, and securing funding and support from ASEAN's external partners. City diplomacy in the context of a smart city is highly stimulated by knowledge, creativity, and innovations[58]

The European Union (EU) has devoted constant efforts to devising a strategy for achieving 'smart' urban growth for its metropolitan city-regions.[59][60] The EU has developed a range of programmes under "Europe's Digital Agenda".[61] In 2010, it highlighted its focus on strengthening innovation and investment in ICT services for the purpose of improving public services and quality of life.[60] Arup estimates that the global market for smart urban services will be $400 billion per annum by 2020.[62]

The Smart Cities Mission is a retrofitting and urban renewal program being spearheaded by the Ministry of Urban Development, Government of India. The Government of India has the ambitious vision of developing 100 cities by modernizing existing mid-sized cities.[63]
Technologies[edit]

Smart grids are an important technology in smart cities. The improved flexibility of the smart grid permits greater penetration of highly variable renewable energy sources such as solar power and wind power.

Mobile devices (such as smartphones, tablets, ...) are another key technology allowing citizens to connect to the smart city services.[64][65][66]

Smart cities also rely on smart homes and specifically, the technology used in them.[67][68][69][70][71]

Bicycle-sharing systems are an important element in smart cities.[72]

Smart mobility is also important to smart cities.[73]

Intelligent transportation systems and CCTV[74] systems are also being developed.

Some smart cities also have digital libraries.[75][76][77][78]

Online collaborative sensor data management platforms are on-line database services that allow sensor owners to register and connect their devices to feed data into an on-line database for storage and allow developers to connect to the database and build their own applications based on that data.[79][80]

Additional supporting technology include telecommuting,[81][82][83] telehealth,[84][85] the blockchain,[86][87] fintech,[88] online banking technology,[89] ...

Electronic cards (known as smart cards) are another common component in smart city contexts. These cards possess a unique encrypted identifier that allows the owner to log into a range of government provided services (or e-services) without setting up multiple accounts. The single identifier allows governments to aggregate data about citizens and their preferences to improve the provision of services and to determine common interests of groups. This technology has been implemented in Southampton.[11]

Retractable bollards allow to restrict access inside city centers (i.e. to delivery trucks resupplying outlet stores, ...). Opening and closing of such barriers is traditionally done manually, through an electronic pass[90] but can even be done by means of ANPR camera's connected to the bollard system.[91]
Cost-benefit analysis of smart city technologies[edit]

Cost-benefit analysis has been done into smart cities and the individual technologies. These can help to assess whether it is economically and ecologically beneficial to implement some technologies at all, and also compare the cost-effectiveness of each technology among each other[92][93][94][95]
Commercialisation[edit]

Large IT, telecommunication and energy management companies such as Baidu, Alibaba, Tencent, Huawei, Google, Microsoft, Cisco, IBM, and Schneider Electric launched market initiatives for intelligent cities.

    Baidu is working on Apollo, a self-driving technology[96]
    Alibaba has created the City Brain[97][98]
    Tencent is working on medical technology,[96] such as WeChat Intelligent Healthcare, Tencent Doctorwork, and AI Medical Innovation System (AIMIS)[99]
    Huawei has its Safe City Compact Solution which focuses on improving safety in cities[100][101][102]
    Google's subsidiary Sidewalk Labs is focusing on smart cities
    Microsoft has CityNext[103]
    Cisco, launched the global "Intelligent Urbanization" initiative[104] to help cities using the network as the fourth utility for integrated city management, better quality of life for citizens, and economic development.
    IBM announced its Smarter Cities Challenge[105] to stimulate economic growth and quality of life in cities and metropolitan areas with the activation of new approaches of thinking and acting in the urban ecosystem.
    Schneider Electric is working on EcoStruxure[106][107]
    Sensor developers and startup companies[clarification needed] are also continually developing new smart city applications.

Adoption[edit]

Examples of smart city technologies and programs have been implemented in Singapore,[108] India,[109][110] Dubai,[111] Milton Keynes,[112] Southampton,[113] Amsterdam,[30] Barcelona,[114] Madrid,[115] Stockholm,[116] Copenhagen, China,[117] and New York.[118]

Major strategies and achievements related to the spatial intelligence of cities are listed in the Intelligent Community Forum awards from 1999 to 2010, in the cities of Songdo and Suwon (South Korea), Stockholm (Sweden), Gangnam District of Seoul (South Korea), Waterloo, Ontario (Canada), Taipei (Taiwan), Mitaka (Japan), Glasgow (Scotland, UK), Calgary (Alberta, Canada), Seoul (South Korea), New York City (US), LaGrange, Georgia (US), and Singapore, which were recognized for their efforts in developing broadband networks and e-services sustaining innovation ecosystems, growth, and inclusion.[119] There are a number of cities actively pursuing a smart city strategy:
Amsterdam[edit]

Street lamps in Amsterdam have been upgraded to allow municipal councils to dim the lights based on pedestrian usage.[120]

The Amsterdam smart city initiative[30] which began in 2009 currently includes 170+ projects collaboratively developed by local residents, government and businesses.[12] These projects run on an interconnected platform through wireless devices to enhance the city's real-time decision making abilities. The City of Amsterdam (City) claims the purpose of the projects is to reduce traffic, save energy and improve public safety.[121] To promote efforts from local residents, the City runs the Amsterdam Smart City Challenge annually, accepting proposals for applications and developments that fit within the city's framework.[122] An example of a resident developed app is Mobypark, which allows owners of parking spaces to rent them out to people for a fee.[123] The data generated from this app can then be used by the city to determine parking demand and traffic flows in Amsterdam. A number of homes have also been provided with smart energy meters, with incentives provided to those that actively reduce energy consumption.[7][124] Other initiatives include flexible street lighting (smart lighting)[125] which allows municipalities to control the brightness of street lights, and smart traffic management[126] where traffic is monitored in real time by the city and information about current travel time on certain roads is broadcast to allow motorists to determine the best routes to take.
Barcelona[edit]

Barcelona has established a number of projects that can be considered 'smart city' applications within its "CityOS" strategy.[127] For example, sensor technology has been implemented in the irrigation system in Parc del Centre de Poblenou, where real time data is transmitted to gardening crews about the level of water required for the plants.[114][128] Barcelona has also designed a new bus network based on data analysis of the most common traffic flows in Barcelona, utilising primarily vertical, horizontal and diagonal routes with a number of interchanges.[129] Integration of multiple smart city technologies can be seen through the implementation of smart traffic lights[130] as buses run on routes designed to optimise the number of green lights. In addition, where an emergency is reported in Barcelona, the approximate route of the emergency vehicle is entered into the traffic light system, setting all the lights to green as the vehicle approaches through a mix of GPS and traffic management software, allowing emergency services to reach the incident without delay. Much of this data is managed by the Sentilo Platform.[131][132]
Columbus, Ohio[edit]

In the summer of 2017, the City of Columbus, Ohio began its pursuit of a smart city initiative. The city partnered with American Electric Power Ohio to create a group of new electric vehicle charging stations. Many smart cities such as Columbus are using agreements such as this one to prepare for climate change, expand electric infrastructure, convert existing public vehicle fleets to electric cars, and create incentives for people to share rides when commuting. For doing this, the U.S. Department of Transportation gave the City of Columbus a $40 million grant. The city also received $10 million from Vulcan Inc.[133]

One key reason why the utility was involved in the picking of locations for new electric vehicle charging stations was to gather data. According to Daily Energy Insider, the group Infrastructure and Business Continuity for AEP said, "You don't want to put infrastructure where it won't be used or maintained. The data we collect will help us build a much bigger market in the future."[133]

Because autonomous vehicles are currently seeing "an increased industrial research and legislative push globally", building routes and connections for them is another important part of the Columbus smart city initiative.[133]
Copenhagen[edit]

In 2014, Copenhagen claimed the prestigious World Smart Cities Award for its “Connecting Copenhagen” smart city development strategy.[134] Positioned in the Technical and Environmental Administration of Copenhagen, the smart city initiatives are coordinated by Copenhagen Solutions Lab, the city's administrative unit for smart city development. There are other notable actors in Greater Copenhagen that coordinate and initiate smart city initiatives including State of Green and Gate21, the latter of which has initiated the innovation hub smart city Cluster Denmark.

In an article with The Economist,[135] a current major smart city project is explained: “In Copenhagen, as in many cities around the world, air quality is high on the agenda when it comes to liveability, with 68 percent of citizens citing it as of high importance when it comes to what makes their city attractive. To monitor pollution levels, Copenhagen Solutions Lab is currently working with Google and has installed monitoring equipment in their streetview car in order to produce a heatmap of air quality around the city. The information will help cyclists and joggers plan routes with the best air quality. The project also gives a glimpse of the future, when this kind of information could be collected in real time by sensors all over the city and collated with traffic flow data.”

In another article with The World Economic Forum, Marius Sylvestersen, Program Director at Copenhagen Solutions Lab, explains that public-private collaborations must be built on transparency, the willingness to share data and must be driven by the same set of values. This requires a particularly open mindset from the organisations that wish to get involved. To facilitate open collaboration and knowledge-sharing, Copenhagen Solutions Lab launched the Copenhagen Street Lab in 2016. Here, organisations such as TDC, Citelum and Cisco work in collaboration with Copenhagen Solutions Lab to identify new solutions to city and citizen problems.
Dubai[edit]

In 2013, the Smart Dubai project was initiated by Shaikh Mohammad bin Rashid Al Maktoum, vice president of UAE, which contained more than 100 initiatives to make Dubai a smart city by 2030. The project aimed to integrate private and public sectors, enabling citizens to access these sectors through their smartphones. Some initiatives include the Dubai Autonomous Transportation Strategy to create driverless transits, fully digitizing government, business and customer information and transactions, and providing citizens 5000 hotspots to access government applications by 2021.[136][137] Two mobile applications, mPay and DubaiNow, facilitate various payment services for citizens ranging from utilities or traffic fines to educational, health, transport, and business services. In addition, the Smart Nol Card is a unified rechargeable card enabling citizens to pay for all transportation services such as metro, buses, water bus, and taxis. There is also the Dubai Municipality's Digital City initiative which assigns each building a unique QR code that citizens can scan containing information about the building, plot, and location.[138]
Dublin[edit]

Dublin has been refereed to as an unexpected capital for smart cities.[139] The smart city programme for the city is run by Smart Dublin[140] an initiative of the four Dublin Local Authorities to engage with smart technology providers, researchers and citizens to solve city challenges and improve city life. It includes Dublinked- Dublin's open data platform that hosts open source data to smart city applications.
Gdynia[edit]

Gdynia was the first city in Eastern Europe to receive the ISO 37120 certificate issued by the World Council on City Data.[141][142] In 2015, the TRISTAR intelligent road traffic management system was implemented in the city.[143] Trolleybuses in Gdynia have been operating since 1943 and are still being developed as low-emission transport - some of them have their own batteries, which allows them to reach areas with no traction.[144][145]

Over 200 sets of up-to-date data from 21 areas of the city's functioning are published on the Open Data portal. The data sets meet the requirements of machine readability and are also presented in a way comprehensible to users.[146] There is also an Urban Lab for cooperation between residents, experts and representatives of city structures.[147][148][149]
Isfahan[edit]

Isfahan has a smart city program, a unified human resources administration system, transport system[150][151].[152][153][154]
Kyiv[edit]

Kyiv has a transport dispatch system. It contains GPS trackers, installed on public transportation, as well as 6,000 video surveillance cameras which monitor the traffic. The accrued data is used by local Traffic Management Service and transport application developers.
London[edit]

In London, a traffic management system known as SCOOT optimises green light time at traffic intersections by feeding back magnetometer and inductive loop data to a supercomputer, which can co-ordinate traffic lights across the city to improve traffic throughout.[155]
Madrid[edit]

Madrid, Spain's pioneering smart city,[156] has adopted the MiNT Madrid Inteligente/Smarter Madrid platform to integrate the management of local services. These include the sustainable and computerized management of infrastructure, garbage collection and recycling, and public spaces and green areas, among others.[157] The programme is run in partnership with IBMs INSA, making use of the latter's Big Data and analytics capabilities and experience.[158] Madrid is considered to have taken a bottom-up approach to smart cities, whereby social issues are first identified and individual technologies or networks are then identified to address these issues.[159] This approach includes support and recognition for start ups through the Madrid Digital Start Up programme.[160]
Malta[edit]

A document written in 2011 refers to 18th century Żejtun as the earliest "smart city" in Malta,[161] but not in the modern context of a smart city. By the 21st century, SmartCity Malta, a planned technology park, became partially operational while the rest is under construction, as a Foreign Direct Investment.
Manchester[edit]

In December 2015, Manchester's CityVerve project was chosen as the winner of a government-led technology competition and awarded £10m to develop an Internet of Things (IoT) smart cities demonstrator.[162]

Established in July 2016, the project is being carried out by a consortium of 22 public and private organisations, including Manchester City Council, and is aligned with the city's on-going devolution commitment.[163]

The project has a two-year remit to demonstrate the capability of IoT applications and address barriers to deploying smart cities, such as city governance, network security, user trust and adoption, interoperability, scalability and justifying investment.

CityVerve is based on an open data principle that incorporates a "platform of platforms"[164] which ties together applications for its four key themes: transport and travel; health and social care; energy and the environment; culture and the public realm. This will also ensure that the project is scalable and able to be redeployed to other locations worldwide.
Milan[edit]

Milan, Italy was prompted to begin its smart city strategies and initiatives by the European Union's Smart Cities and Communities initiative. However, unlike many European cities, Milan's Smart city strategies focus more on social sustainability rather than environmental sustainability.[165] This focus is almost exclusive to Milan and has a major influence in the way content and way its strategies are implemented as shown in the case study of the Bicocca District in Milan.[166]
Milton Keynes[edit]

Milton Keynes has a commitment to making itself a smart city. Currently the mechanism through which this is approached is the MK:Smart initiative,[112] a collaboration of local government, businesses, academia and 3rd sector organisations. The focus of the initiative is on making energy use, water use and transport more sustainable whilst promoting economic growth in the city. Central to the project is the creation of a state-of-the-art 'MK Data Hub' which will support the acquisition and management of vast amounts of data relevant to city systems from a variety of data sources. These will include data about energy and water consumption, transport data, data acquired through satellite technology, social and economic datasets, and crowdsourced data from social media or specialised apps.

The MK:Smart initiative has two aspects which extend our understanding of how smart Cities should operate. The first, Our MK,[167] is a scheme for promoting citizen-led sustainability issues in the city. The scheme provides funding and support to engage with citizens and help turn their ideas around sustainability into a reality. The second aspect is in providing citizens with the skills to operate effectively in a smart city. The Urban Data school[168] is an online platform to teach school students about data skills while the project has also produced a MOOC[169] to inform citizens about what a smart city is.
Moscow[edit]

Moscow has been implementing smart solutions since 2011 by creating the main infrastructure and local networks. Over the past few years Moscow Government implemented a number of programs, contributing to its IT development. So, Information City programme was launched and subsequently implemented from 2012 to 2018. The initial purpose of the programme was to make daily life for citizens safe and comfortable through the large-scale introduction of information and communication technologies.[170] In the summer of 2018, Moscow Mayor Sergey Sobyanin announced the city's smart city project, aimed at applying modern technologies in all areas of city life.[171] And in June 2018, the global management consultancy McKinsey announced that Moscow is one of the world's top 50 cities for smart technologies.[172] Smart City technologies have been deployed in healthcare, education, transport and municipal services. The initiative aims to improve quality of life, make urban government more efficient and develop an information society. There are more than 300 digital initiatives within the smart city project, with electronic services now widely provided online and through multifunctional centres. Moscow's citywide Wi-Fi project was launched in 2012 and now provides more than 16,000 Wi-Fi internet access points.[173] The total number of access points will exceed 20,500 by early 2021.[174] Moscow is actively developing eco-friendly transport using electric buses, and autonomous cars will soon be tested on the city's streets. Other initiatives include Moscow's Electronic School programme, its blockchain-based Active Citizen project and smart traffic management.[171]
New Songdo City[edit]

[175][176]
New York[edit]

New York is developing a number of smart city initiatives. An example is the series of city service kiosks in the LinkNYC network. These provide services including free WiFi, phone calls, device charging stations, local wayfinding, and more, funded by advertising that plays on the kiosk's screens.[177]
San Leandro[edit]

The city of San Leandro, California is in the midst of transforming from an industrial center to a tech hub of the Internet of things (IoT) (technology that lets devices communicate with each other over the Internet). California's utility company PG&E is working with the city in this endeavor and on a smart energy pilot program that would develop a distributed energy network across the city that would be monitored by IoT sensors. The goal would be to give the city an energy system that has enough capacity to receive and redistribute electricity to and from multiple energy sources.[52]
Santa Cruz[edit]

An alternative use of smart city technology can be found in Santa Cruz, California, where local authorities analyse historical crime data in order to predict police requirements and maximise police presence where it is required.[178] The analytical tools generate a list of 10 places each day where property crimes are more likely to occur, and then placing police efforts on these regions when officers are not responding to any emergency. This use of ICT technology is different to the manner in which European cities utilise smart city technology, possibly highlighting the breadth of the smart city concept in different parts of the world.
Santander[edit]

The city of Santander in Cantabria, northern Spain, has 20,000 sensors connecting buildings, infrastructure, transport, networks and utilities, offers a physical space for experimentation and validation of the IoT functions, such as interaction and management protocols, device technologies, and support services such as discovery, identity management and security[179] In Santander, the sensors monitor the levels of pollution, noise, traffic and parking.
Shanghai[edit]

Shanghai's development of the IoT and internet connection speeds have allowed for third-party companies to revolutionize the productivity of the city.[180] As mobile ride share giant, DiDi Chuxing, continuously adds more user protection features such as ride recording, and a new quick response safety center, Shanghai is furthering their smart city agenda.[181] During the first China International Import Expo, Shanghai focused on smart mobility and implemented sensors to accept smartphone traffic cards in all metro stations and buses to increase efficiency in the city.
Singapore[edit]

Singapore, a city-state, has embarked on transforming towards a "Smart Nation", and endeavours to harness the power of networks, data and info-comm technologies to improve living, create economic opportunities and build closer communities.
Stockholm[edit]

The Kista Science City from above.

Stockholm's smart city technology is underpinned by the Stokab dark fibre system[182] which was developed in 1994 to provide a universal fibre optic network across Stockholm.[183] Private companies are able to lease fibre as service providers on equal terms. The company is owned by the City of Stockholm itself.[116] Within this framework, Stockholm has created a Green IT strategy.[184] The Green IT program seeks to reduce the environmental impact of Stockholm through IT functions such as energy efficient buildings (minimising heating costs), traffic monitoring (minimising the time spent on the road) and development of e-services (minimising paper usage). The e-Stockholm platform is centred on the provision of e-services, including political announcements, parking space booking and snow clearance.[185] This is further being developed through GPS analytics, allowing residents to plan their route through the city.[185] An example of district-specific smart city technology can be found in the Kista Science City region.[186] This region is based on the triple helix concept of smart cities,[21] where university, industry and government work together to develop ICT applications for implementation in a smart city strategy.
Taipei[edit]

Taipei started the "smarttaipei" project in 2016, where the major concept of is to change the culture of city hall government to be able to adopt new ideas and new concepts called bottom-up mechanism. The Taipei City government established the “Taipei Smart City Project Management Office”, also known as the "PMO", to implement and governance the development of smart city. Thereafter, building an innovation matchmaking platform to combine industry and government resources to develop smart solutions that satisfy public demands.

PMO accept proposals from industry and help to negotiate with relative department of Taipei city to initiate new proof of concept(PoC) project, with the help of a matchmaking platform which allows citizens access necessary innovative technologies. There are more than 150[187] PoC Project established, and only 34% project finished.
Research[edit]

University research labs developed prototypes for intelligent cities.

    IGLUS is an action research project led by EPFL focused on developing governance systems for urban infrastructures. IGLUS announced a MOOC through Coursera.[188]
    MIT Smart Cities Lab[189] focuses upon intelligent, sustainable buildings, mobility systems (GreenWheel electric bicycle, mobility on demand, CityCar, Wheel Robots);
    the IntelCities[190] research consortium for electronic government, planning systems and citizen participation; URENIO developed intelligent city platforms for the innovation economy[191] focusing on strategic intelligence, technology transfer, collaborative innovation, and incubation, while it promotes intelligent cities research and planning;[192]
    the Smart Cities Academic Network[193] is working on e-governance and e-services in the North Sea region.
    The MK:Smart project[112] is focusing on issues of sustainable energy use, water use and transport infrastructure alongside exploring how to promote citizen engagement[167] alongside educating citizens about smart cities.[168][169]
    Laboratory for AI, Machine Learning, Business & Data Analytics (LAMBDA) at Tel Aviv University focuses on Digital Life, Smart Transportation and Human Mobility Patterns in smart Cities.[194]
    Research journals in this area include the UK IET Smart Cities, which was launched in 2018.[195]

Criticism[edit]

The criticisms of smart cities revolve around:[21]

    A bias in strategic interest may lead to ignoring alternative avenues of promising urban development.[196]
    A smart city, as a scientifically planned city, would defy the fact that real development in cities is often haphazard. In that line of criticism, the smart city is seen as unattractive for citizens as they "can deaden and stupefy the people who live in its all-efficient embrace".[197] Instead, people would prefer cities they can participate to shape.
    The focus of the concept of smart city may lead to an underestimation of the possible negative effects of the development of the new technological and networked infrastructures needed for a city to be smart.[198]
    As a globalized business model is based on capital mobility, following a business-oriented model may result in a losing long-term strategy: "The 'spatial fix' inevitably means that mobile capital can often 'write its own deals' to come to town, only to move on when it receives a better deal elsewhere. This is no less true for the smart city than it was for the industrial, [or] manufacturing city."[21]
    The high level of big data collection and analytics has raised questions regarding surveillance in smart cities, particularly as it relates to predictive policing.
    As of August 2018, the discussion on smart cities centres around the usage and implementation of technology rather than on the inhabitants of the cities and how they can be involved in the process.[199]
    Especially in low-income countries, smart cities are irrelevant to the majority of the urban population, which lives in poverty with limited access to basic services. A focus on smart cities may worsen inequality and marginalization.[200]
    If a smart city strategy is not planned taking into account people with accessibility problems, such as persons with disabilities affecting mobility, vision, hearing, and cognitive function, the implementation of new technologies could create new barriers.[201]""",
}

#CHUNKING

df = pd.DataFrame()

for id in sample_data:
    value = sample_data[id]
    df_this_text = process_text(value)
    df_this_text['id'] = id
    df = pd.concat([df,df_this_text]).reset_index(drop=True)

#INFERENCE

df['sdg'] = process_list_of_text(df.text, model, tokenizer, device)

for i in range(17):
    df["sdg_%i"%(i+1)] = df["sdg"].apply(lambda x: x[i])

df['id_copy'] = df['id']
df_agg = df.groupby([
                    'id',
                    'error',
                    ]
                )["id_copy"].count().reset_index().rename({
                    "id_copy": "num_chunks",
                    "error": "parsing_error"
                    }, axis=1)
df_agg["num_valid_chunks"] = 0
df_agg["document_top_sdg"] = ""

for c in SDG_COLS:
    df_agg[c] = 0

for uuid in tqdm(df.id.unique()):
    df_sdg_smooth = smoothen_sdg_values(df[df.id==uuid][SDG_COLS], window_size=5)
    scores, top_sdg_index, num_valid_chunks = aggregated_sdg_score(df_sdg_smooth, CONFIDENCE_LEVEL=0.5)
    df_agg.loc[df_agg.id==uuid, SDG_COLS] = scores
    df_agg.loc[df_agg.id==uuid, "document_top_sdg"] = SDG_GOALS[top_sdg_index]
    df_agg.loc[df_agg.id==uuid, "num_valid_chunks"] = num_valid_chunks

print(df_agg.head())

#Save results
#df_agg.to_csv(outfile.csv,sep=';', index=False)
