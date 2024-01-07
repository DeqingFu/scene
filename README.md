# SCENE: Self-Labeled Counterfactuals for Extrapolating to Negative Examples

This is an official implementation for our paper, [SCENE: Self-Labeled Counterfactuals for Extrapolating to Negative Examples](https://arxiv.org/abs/2305.07984), EMNLP 2023.

```bibtex
@inproceedings{fu-etal-2023-scene,
    title = "{SCENE}: Self-Labeled Counterfactuals for Extrapolating to Negative Examples",
    author = "Fu, Deqing and Godbole, Ameya and Jia, Robin",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.485",
    doi = "10.18653/v1/2023.emnlp-main.485",
    pages = "7832--7848",
}
```

## Getting Started
1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate scene
    ```

2. Unzip preprocessed retrieval data from SQuAD
   ```
   unzip data.zip
   ```
   The folder ``data`` also includes all synthetic examples generated by SCENE.

3. Run training scripts
- Starting from SQuAD 1.1
   ```
   bash run_squad.sh
   ```
- Starting from SQuAD 2.0
  ```
  bash run_squad_v2.sh
  ```

## SCENE generated examples 
Under ``data/scene.csv`` are all synthetic examples generated by SCENE during the training processes. 
Here are a randomly sampled subset of these generated examples.

|context                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |question                                                                                                       |answer                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------|
|The west end of these streets is Bowery and Third Avenue, except for 3rd Street (formerly Amity Place; to Sixth Avenue) and 4th Street (to 13th Street), which extend west and north, respectively, into Greenwich Village. Great Jones Street connects East 3rd to West 3rd.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |Which team East Coast to pick 3rd?                                                                             |No-Answer                                             |
|The law of the United States comprises many levels of codified and uncodified forms of law, of which the most important is the United States Constitution, the foundation of the federal government of the United States. The Constitution sets out the boundaries of federal law, which consists of acts of Congress, treaties ratified by the Senate, regulations promulgated by the executive branch, and case law originating from the federal judiciary. The United States Code is the official compilation and codification of general and permanent federal statutory law.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |What is the most important document in the US laws?                                                      |the United States Constitution                  |
|In 1839, Melbourne resigned after Radicals and Tories (both of whom Victoria detested) voted against a bill to suspend the constitution of Jamaica. The bill removed political power from plantation owners who were resisting measures associated with the abolition of slavery. The Queen commissioned a Tory, Sir Robert Peel, to form a new ministry. At the time, it was customary for the prime minister to appoint members of the Royal Household, who were usually his political allies and their spouses. Many of the Queen's ladies of the bedchamber were wives of Whigs, and Peel expected to replace them with wives of Tories. In what became known as the bedchamber crisis, Victoria, advised by Melbourne, objected to their removal. Peel refused to govern under the restrictions imposed by the Queen, and consequently resigned his commission, allowing Melbourne to return to office.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |Which party was Sir Anthony Eden once a part of?                                                               |Radicals and Tories                             |
|Though Jehovah's Witnesses do not accept blood transfusions of whole blood, they may accept some blood plasma fractions at their own discretion. The Watch Tower Society provides pre-formatted durable power of attorney documents prohibiting major blood components, in which members can specify which allowable fractions and treatments they will personally accept. Jehovah's Witnesses have established Hospital Liaison Committees as a cooperative arrangement between individual Jehovah's Witnesses and medical professionals and hospitals.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |What types of durable power of attorney documents are currently allowed by the Watch Tower Society to prohibit?|No-Answer                                             |
|In 2005, seventeen countries produced concentrated uranium oxides, with Canada (27.9% of world production) and Australia (22.8%) being the largest producers and Kazakhstan (10.5%), Russia (8.0%), Namibia (7.5%), Niger (7.4%), Uzbekistan (5.5%), the United States (2.5%), Argentina (2.1%), Ukraine (1.9%) and China (1.7%) also producing significant amounts. Kazakhstan continues to increase production and may have become the world's largest producer of uranium by 2009 with an expected production of 12,826 tonnes, compared to Canada with 11,100 t and Australia with 9,430 t. In the late 1960s, UN geologists also discovered major uranium deposits and other rare mineral reserves in Somalia. The find was the largest of its kind, with industry experts estimating the deposits at over 25% of the world's then known uranium reserves of 800,000 tons.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |As of 2017, what country was the largest producer and exporter of uranium oxides?                              |No-Answer                                             |
|In August 1836, two real estate entrepreneurs—Augustus Chapman Allen and John Kirby Allen—from New York, purchased 6,642 acres (26.88 km2) of land along Buffalo Bayou with the intent of founding a city. The Allen brothers decided to name the city after Sam Houston, the popular general at the Battle of San Jacinto, who was elected President of Texas in September 1836. The great majority of slaves in Texas came with their owners from the older slave states. Sizable numbers, however, came through the domestic slave trade. New Orleans was the center of this trade in the Deep South, but there were slave dealers in Houston. Thousands of enslaved African-Americans lived near the city before the Civil War. Many of them near the city worked on sugar and cotton plantations, while most of those in the city limits had domestic and artisan jobs. In 1860 forty-nine percent of the city's population was enslaved. A few slaves, perhaps as many as 2,000 between 1835 and 1865, came through the illegal African trade. Post-war Texas grew rapidly as migrants poured into the cotton lands of the state. They also brought or purchased enslaved African Americans, whose numbers nearly tripled in the state from 1850 to 1860, from 58,000 to 182,566.                                                                                                    |Why was Sam Houston elected President of Texas?                                                                |the popular general at the Battle of San Jacinto|
|ADSL-broadband service is provided with maximum speeds of up to 1536 KBit/s downstream and 512 KBit/s upstream offered on contract levels from lite £16 per month to gold+ at £190 per month. There are a few public WiFi hotspots in Jamestown, which are also being operated by SURE (formerly Cable & Wireless).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |What location has wifi for public?                                                                             |Jamestown                                       |
|Japan used the name Greater East Asia War (大東亜戦争, Dai Tō-A Sensō?), as chosen by a cabinet decision on 10 December 1941, to refer to both the war with the Western Allies and the ongoing war in China. This name was released to the public on 12 December, with an explanation that it involved Asian nations achieving their independence from the Western powers through armed forces of the Greater East Asia Co-Prosperity Sphere. Japanese officials integrated what they called the Japan–China Incident (日支事変, Nisshi Jihen?) into the Greater East Asia War.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |By what name was the treaty of friendship with Japan referred?                                                 |No-Answer                                             |
|Universal Studios Inc. (also known as Universal Pictures) is an American film studio, owned by Comcast through its wholly owned subsidiary NBCUniversal, and is one of Hollywood's "Big Six" film studios. Its production studios are at 100 Universal City Plaza Drive in Universal City, California. Distribution and other corporate offices are in New York City. Universal Studios is a member of the Motion Picture Association of America (MPAA). Universal was founded in 1912 by the German Carl Laemmle (pronounced "LEM-lee"), Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |How was Universal Studios created?                                                                             |No-Answer                                             |
|Honors and tributes flowed to Bell in increasing numbers as his most famous invention became ubiquitous and his personal fame grew. Bell received numerous honorary degrees from colleges and universities, to the point that the requests almost became burdensome. During his life he also received dozens of major awards, medals and other tributes. These included statuary monuments to both him and the new form of communication his telephone created, notably the Bell Telephone Memorial erected in his honor in Alexander Graham Bell Gardens in Brantford, Ontario, in 1917.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |When will the Bell Telephone Memorial constructed?                                                             |No-Answer                                             |
|A census of sea life carried out during the International Polar Year and which involved some 500 researchers was released in 2010. The research is part of the global Census of Marine Life (CoML) and has disclosed some remarkable findings. More than 235 marine organisms live in both polar regions, having bridged the gap of 12,000 km (7,456 mi). Large animals such as some cetaceans and birds make the round trip annually. More surprising are small forms of life such as mudworms, sea cucumbers and free-swimming snails found in both polar oceans. Various factors may aid in their distribution – fairly uniform temperatures of the deep ocean at the poles and the equator which differ by no more than 5 °C, and the major current systems or marine conveyor belt which transport eggs and larval stages.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |How many of these animals live in the polar regions?                                                           |235 marine organisms                            |
|The Slavic autonym *Slověninъ is usually considered (e.g. by Roman Jakobson) a derivation from slovo "word", originally denoting "people who speak (the same language)," i.e. people who understand each other, in contrast to the Slavic word denoting "foreign people" – němci, meaning "mumbling, murmuring people" (from Slavic *němъ – "mumbling, mute").                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |What slavic word means "foreign people?                                                                        |němci                                           |
|Spanish explorer Alonso de Salazar was the first European to see the islands in 1526, commanding the ship Santa Maria de la Victoria, the only surviving vessel of the Loaísa Expedition. On August 21, he sighted an island (probably Taongi) at 14°N that he named "San Bartolome".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |What will Salazar most likely see?                                                                             |No-Answer                                             |
|A further line in the directive stressed the need to inflict the heaviest losses possible, but also to intensify the air war in order to create the impression an amphibious assault on Britain was planned for 1941. However, meteorological conditions over Britain were not favourable for flying and prevented an escalation in air operations. Airfields became water-logged and the 18 Kampfgruppen (bomber groups) of the Luftwaffe's Kampfgeschwadern (bomber wings) were relocated to Germany for rest and re-equipment.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |What is preventing the resumption of air operations?                                                           |No-Answer                                             |
|Subjective idealists like George Berkeley are anti-realists in terms of a mind-independent world, whereas transcendental idealists like Immanuel Kant are strong skeptics of such a world, affirming epistemological and not metaphysical idealism. Thus Kant defines idealism as "the assertion that we can never be certain whether all of our putative outer experience is not mere imagining". He claimed that, according to idealism, "the reality of external objects does not admit of strict proof. On the contrary, however, the reality of the object of our internal sense (of myself and state) is clear immediately through consciousness."  However, not all idealists restrict the real or the knowable to our immediate subjective experience. Objective idealists make claims about a transempirical world, but simply deny that this world is essentially divorced from or ontologically prior to the mental. Thus Plato and Gottfried Leibniz affirm an objective and knowable reality transcending our subjective awareness—a rejection of epistemological idealism—but propose that this reality is grounded in ideal entities, a form of metaphysical idealism. Nor do all metaphysical idealists agree on the nature of the ideal; for Plato, the fundamental entities were non-mental abstract forms, while for Leibniz they were proto-mental and concrete monads.|A famous philosopher was a transcendental idealist?                                                            |Immanuel Kant                                   |
|Spielberg then revisited his Close Encounters project and, with financial backing from Columbia Pictures, released Close Encounters: The Special Edition in 1980. For this, Spielberg fixed some of the flaws he thought impeded the original 1977 version of the film and also, at the behest of Columbia, and as a condition of Spielberg revising the film, shot additional footage showing the audience the interior of the mothership seen at the end of the film (a decision Spielberg would later regret as he felt the interior of the mothership should have remained a mystery). Nevertheless, the re-release was a moderate success, while the 2001 DVD release of the film restored the original ending.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |How much studio funded Close Encounters?                                                                       |Columbia Pictures                               |
