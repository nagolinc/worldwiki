from chat_openai import chat
import json
import random
import dataset
import time

import argparse
from collections import Counter


# Connect to the database
db = dataset.connect('sqlite:///stories.db')

# Create tables
stories_table = db['stories']
entities_table = db['entities']
edges_table = db['edges']

def expand(item, entity=None, top_k=5):
    if entity is None:
        #we need to retrieve all edges for this story
        # and randomly choose a new entity
        # we can do this by querying the edges_table
        # for all rows with story_id equal to item["id"]
        # and then randomly choosing one of the entity_name values
        # from the resulting rows
        edges = list(edges_table.find(story_id=item["id"]))
        entity_name = random.choice(edges)["entity_name"]
        #retrieve the entity from the entities_table
        entity = entities_table.find_one(name=entity_name)

    print(entity)

    history = [{"role": "system", "content": system_prompt_write_story}]

    # Fetch neighbors
    nearest = pagerank_like_algorithm(item["title"], entity["name"], iterations=10)
    print("Title:", item["title"])
    print("Nearest:")
    for n in nearest[top_k::-1]:
        print(n)
        # Add to history
        history += [
            {"role": "user", "content": stories_table.find_one(title=n[0])["prompt"]},
            {"role": "assistant", "content": stories_table.find_one(title=n[0])["story"]},
        ]

    questions, _ = chat(item["story"] + "\nEntity: " + entity["name"], system_prompt_generate_questions)

    print(questions)

    # Choose a question and ask it
    question = random.choice(questions.split("\n"))

    print("\n\nQuestion:", question, "\n\n")

    history += [
        {"role": "user", "content": item["prompt"]},
        {"role": "system", "content": item["story"]},
    ]

    story, _ = chat(question + reminders_write_story, history=history)

    print(story)

    history_for_extract = [{"role": "system", "content": system_prompt_extract_entities}]
    for title, score in nearest:
        thisitem = stories_table.find_one(title=title)
        #we need to add linked entities to this item
        edges = list(edges_table.find(story_id=thisitem["id"]))
        entities = [entities_table.find_one(name=edge["entity_name"],story_id=edge["story_id"]) for edge in edges]
        thisitem["entities"] = entities


        thisstory = thisitem["story"]
        # Remove entity, story, prompt from item and convert to json
        thisJson = json.dumps({k: v for k, v in thisitem.items() if k not in ["entity", "story", "prompt"]})
        history_for_extract += [
            {"role": "user", "content": thisstory},
            {"role": "assistant", "content": thisJson},
        ]

    _entities, _ = chat(story + reminders_extract_entities, system_prompt_extract_entities, history=history_for_extract, json_mode=True)

    print(_entities)

    newItem = json.loads(_entities)

    #verify that newItem["title"] is unique
    title=newItem["title"]
    t=0
    while stories_table.find_one(title=newItem["title"]):
        t+=1
        newItem["title"]=title+str(t)

    newItem["entity"] = entity["name"]
    newItem["story"] = story
    newItem["prompt"] = question
    id = int(time.time() * 1000) + random.randint(0, 999)
    newItem["id"] = id

    # Add to the database
    #we need a filtered version of newItem with only the fields we want to store
    # whcih are id, title, prompt, story
    filteredItem = {k: v for k, v in newItem.items() if k in ["id", "title", "prompt", "story"]}
    stories_table.insert(filteredItem)
    for entity in newItem["entities"]:
        entities_table.insert({"name": entity["name"], 
                               "story_id": newItem["id"],
                               "type": entity["type"],
                               "description": entity["description"],
                               "status": entity["status"],
                               })
        edges_table.insert({"story_id": newItem["id"], "entity_name": entity["name"]})

    return newItem



def connectStories(story1_title, entity1_name, story2_title, entity2_name, top_k=5):
    

    history = [{"role": "system", "content": system_prompt_write_story}]

    # Fetch neighbors (by finding flow from story1 to story2)
    nearest = ford_fulkerson(story1_title, story2_title)
    print("Nearest:")
    for n in nearest[top_k::-1]:
        print(n)
        # Add to history
        history += [
            {"role": "user", "content": stories_table.find_one(title=n[0])["prompt"]},
            {"role": "assistant", "content": stories_table.find_one(title=n[0])["story"]},
        ]

    item1=stories_table.find_one(title=story1_title)
    item2=stories_table.find_one(title=story2_title)

    prompt=f"write 5-10 questions about the connection between {entity1_name} and {entity2_name}"

    questions, _ = chat(prompt, 
                        system_prompt_generate_questions)

    print(questions)

    # Choose a question and ask it
    question = random.choice(questions.split("\n"))

    print("\n\nQuestion:", question, "\n\n")

    story, _ = chat(question + reminders_write_story, history=history)

    print(story)

    history_for_extract = [{"role": "system", "content": system_prompt_extract_entities}]
    for title, score in nearest:
        thisitem = stories_table.find_one(title=title)
        #we need to add linked entities to this item
        edges = list(edges_table.find(story_id=thisitem["id"]))
        entities = [entities_table.find_one(name=edge["entity_name"],story_id=edge["story_id"]) for edge in edges]
        thisitem["entities"] = entities


        thisstory = thisitem["story"]
        # Remove entity, story, prompt from item and convert to json
        thisJson = json.dumps({k: v for k, v in thisitem.items() if k not in ["entity", "story", "prompt"]})
        history_for_extract += [
            {"role": "user", "content": thisstory},
            {"role": "assistant", "content": thisJson},
        ]

    _entities, _ = chat(story + reminders_extract_entities, system_prompt_extract_entities, history=history_for_extract, json_mode=True)

    print(_entities)

    newItem = json.loads(_entities)

    #verify that newItem["title"] is unique
    title=newItem["title"]
    t=0
    while stories_table.find_one(title=newItem["title"]):
        t+=1
        newItem["title"]=title+str(t)

    newItem["entity"] = entity1_name
    newItem["story"] = story
    newItem["prompt"] = question
    id = int(time.time() * 1000) + random.randint(0, 999)
    newItem["id"] = id

    # Add to the database
    #we need a filtered version of newItem with only the fields we want to store
    # whcih are id, title, prompt, story
    filteredItem = {k: v for k, v in newItem.items() if k in ["id", "title", "prompt", "story"]}
    stories_table.insert(filteredItem)
    for entity in newItem["entities"]:
        entities_table.insert({"name": entity["name"], 
                               "story_id": newItem["id"],
                               "type": entity["type"],
                               "description": entity["description"],
                               "status": entity["status"],
                               })
        edges_table.insert({"story_id": newItem["id"], "entity_name": entity["name"]})

    return newItem

def addRandomEdge():
    titles=[x["title"] for x in  stories_table.all()]
    title1,title2=random.sample(titles,2)
    story1=stories_table.find_one(title=title1)
    title1=story1["title"]
    edges1=list(edges_table.find(story_id=story1["id"]))
    entity1=random.choice(edges1)['entity_name']

    story2=stories_table.find_one(title=title2)
    title2=story2["title"]
    edges2=list(edges_table.find(story_id=story2["id"]))
    entity2=random.choice(edges2)['entity_name']

    print("ADDING EDGE",title1,entity1,title2,entity2,sep='\n')

    item=connectStories(title1,entity1,title2,entity2)
    return item


from flask import Flask, request, jsonify
import random

app = Flask(__name__)


@app.route('/get_story', methods=['GET'])
def get_story():
    title = request.args.get('title')
    story = stories_table.find_one(title=title)

    #fetch attached entities
    #first find edges
    edges = list(edges_table.find(story_id=story["id"]))
    #then find entities
    entities = [entities_table.find_one(name=edge["entity_name"],story_id=edge["story_id"]) for edge in edges]

    story["entities"] = entities

    if story:
        return jsonify(story)
    else:
        return jsonify({"error": "Title not found"}), 404


@app.route('/list_entity', methods=['GET'])
def list_entity():
    entity_name = request.args.get('entity')
    stories = [row["story_id"] for row in edges_table.find(entity_name=entity_name)]
    titles = [stories_table.find_one(id=story_id)["title"] for story_id in stories]
    return jsonify(titles)


@app.route('/expand', methods=['GET'])
def expand_story():
    title = request.args.get('title')
    entity_name = request.args.get('entity')
    story = stories_table.find_one(title=title)
    if story:
        entity = entities_table.find_one(name=entity_name, story_id=story["id"])
        if entity:
            expanded_item = expand(story, entity)
            return jsonify(expanded_item)
        else:
            return jsonify({"error": "Entity not found"}), 404
    else:
        return jsonify({"error": "Title not found"}), 404


def pagerank_like_algorithm(initial_item_title, initial_entity_name, iterations=10):
    # Initialize weights
    weights = {row["title"]: 0 for row in stories_table}
    weights[initial_item_title] = 1
    entity_weights = {row["name"]: 0 for row in entities_table}
    entity_weights[initial_entity_name] = 1

    # Create graph representation
    neighbors = {row["title"]: set() for row in stories_table}
    entity_neighbors = {row["name"]: set() for row in entities_table}

    for row in edges_table:
        neighbors[stories_table.find_one(id=row["story_id"])["title"]].add(row["entity_name"])
        entity_neighbors[row["entity_name"]].add(stories_table.find_one(id=row["story_id"])["title"])

    # Weight distribution
    for _ in range(iterations):
        new_weights = {row["title"]: 0 for row in stories_table}
        new_entity_weights = {row["name"]: 0 for row in entities_table}

        for item_title, weight in weights.items():
            if weight > 0:
                num_neighbors = len(neighbors[item_title])
                if num_neighbors > 0:
                    distributed_weight = weight / num_neighbors
                    for neighbor in neighbors[item_title]:
                        new_entity_weights[neighbor] += distributed_weight

        for entity_name, weight in entity_weights.items():
            if weight > 0:
                num_neighbors = len(entity_neighbors[entity_name])
                if num_neighbors > 0:
                    distributed_weight = weight / num_neighbors
                    for neighbor in entity_neighbors[entity_name]:
                        new_weights[neighbor] += distributed_weight

        # Re-add initial weights
        new_weights[initial_item_title] += 1
        new_entity_weights[initial_entity_name] += 1

        weights = new_weights
        entity_weights = new_entity_weights

    # Sort and return items by weights
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


from collections import deque, defaultdict

def bfs_find_path(source, sink, neighbors, capacity, flow):
    queue = deque([source])
    paths = {source: []}
    while queue:
        u = queue.popleft()
        for v in neighbors[u]:
            if v not in paths and capacity[u, v] - flow[u, v] > 0:
                paths[v] = paths[u] + [(u, v)]
                if v == sink:
                    return paths[v]
                queue.append(v)
    return None

def ford_fulkerson(source_story, target_story):
    # Initialize flow
    flow = defaultdict(int)

    # Create graph representation
    neighbors = defaultdict(set)
    capacity = defaultdict(int)

    for row in edges_table:
        story_title = stories_table.find_one(id=row["story_id"])["title"]
        entity_name = row["entity_name"]
        neighbors[story_title].add(entity_name)
        neighbors[entity_name].add(story_title)
        capacity[story_title, entity_name] = 1
        capacity[entity_name, story_title] = 1

    source = source_story
    sink = target_story

    # Ford-Fulkerson algorithm
    max_flow = 0
    while True:
        path = bfs_find_path(source, sink, neighbors, capacity, flow)
        #print(path)
        if not path:
            break
        # Find the maximum flow through the path
        path_flow = min(capacity[u, v] - flow[u, v] for u, v in path)
        for u, v in path:
            flow[u, v] += path_flow
            flow[v, u] -= path_flow
        max_flow += path_flow

    #now we need to find the flow though each story, and sort them by flow
    story_flows_in = {story["title"]: 0 for story in stories_table}
    story_flows_out = {story["title"]: 0 for story in stories_table}
    #print(flow)
    for (u, v), f in flow.items():
        if u in story_flows_in:
            if f > 0:
                story_flows_out[u] += f
            else:
                story_flows_in[u] += abs(f)
        if v in story_flows_in:
            if f > 0:
                story_flows_in[v] += f
            else:
                story_flows_out[v] += abs(f)

    story_flows = {story: max(story_flows_in[story], story_flows_out[story]) for story in story_flows_in}
    
    sorted_stories = sorted(story_flows.items(), key=lambda x: x[1], reverse=True)
    #filter out stories with flow 0
    sorted_stories = [(story, flow) for story, flow in sorted_stories if flow > 0]
    return sorted_stories


# Endpoint to list all entities by name and count of pages linking to each entity
@app.route('/list_entities', methods=['GET'])
def list_entities():
    # Count occurrences of each entity in edges_table
    entity_counts = Counter(row["entity_name"] for row in edges_table)
    
    # Convert to a list of dictionaries
    entities = [{"name": name, "count": count} for name, count in entity_counts.items()]
    
    return jsonify(entities)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the app')

    #tory_intro = "at the beginning of time the god Logochronos spoke the world into being"

    parser.add_argument('--story_intro', type=str, default="at the beginning of time the god Logochronos spoke the world into being",
                        help='The introduction to the story')


    #--system_prompt_write_story
    parser.add_argument('--system_prompt_write_story', type=str, default="system_prompt_write_story.txt",
                        help='The system prompt for writing a story')

    #--reminders_write_story
    parser.add_argument('--reminders_write_story', type=str, default="reminders_write_story.txt",
                        help='The reminders for writing a story')
    
    args = parser.parse_args()

    # Load system prompts
    system_prompt_write_story = open(args.system_prompt_write_story).read()
    reminders_write_story = open(args.reminders_write_story).read()
    system_prompt_extract_entities = open("system_prompt_extract_entities.txt").read()
    reminders_extract_entities = open("reminders_extract_entities.txt").read()
    system_prompt_generate_questions = open("system_prompt_generate_questions.txt").read()


    #check if the database is empty
    if len(stories_table) == 0:

        
        story, _ = chat(args.story_intro, system_prompt_write_story)

        print(story)

        _entities, _ = chat(story, system_prompt_extract_entities, json_mode=True)

        print(_entities)

        startingItem = json.loads(_entities)

        startingItem["story"] = story
        startingItem["prompt"] = args.story_intro
        startingItem["entity"] = startingItem["entities"][0]["name"]

        print("Entity:", startingItem["entity"])

        #make up an id, I guess (how do we make srue it's unique?)
        unique_id = int(time.time() * 1000) + random.randint(0, 999)
        startingItem["id"] = unique_id

        filteredItem = {k: v for k, v in startingItem.items() if k in ["id", "title", "prompt", "story"]}
        stories_table.insert(filteredItem)
        for entity in startingItem["entities"]:
            entities_table.insert({"name": entity["name"], 
                                   "story_id": startingItem["id"],
                                   "type": entity["type"],
                                   "description": entity["description"],
                                   "status": entity["status"],
                                   })
            edges_table.insert({"story_id": startingItem["id"], "entity_name": entity["name"]})

        newItem = expand(startingItem)

        print(newItem)

        num_starting_items = 2

        for i in range(num_starting_items):
            thisItem = random.choice(list(stories_table))
            newItem = expand(thisItem)

    app.run(debug=True, use_reloader=False)

else:
    # Load system prompts
    system_prompt_write_story = open("system_prompt_write_story.txt").read()
    reminders_write_story = open("reminders_write_story.txt").read()
    system_prompt_extract_entities = open("system_prompt_extract_entities.txt").read()
    reminders_extract_entities = open("reminders_extract_entities.txt").read()
    system_prompt_generate_questions = open("system_prompt_generate_questions.txt").read()
