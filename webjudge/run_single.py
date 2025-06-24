import argparse
import os
from webjudge.methods.agenttrek_eval import *
from webjudge.methods.automomous_eval import *
from webjudge.methods.webjudge_general_eval import *
from webjudge.methods.webjudge_online_mind2web import *
from webjudge.methods.webvoyager_eval import *
from webjudge.utils import OpenaiEngine, extract_predication
import json
import copy
import asyncio
import multiprocessing

def extract_step(fname):
    name, _ = os.path.splitext(fname)   # "GENERAL-1826..._1"
    try:
        return int(name.split('_')[-1]) # "_" 뒤 숫자 부분
    except ValueError:
        return float('inf') 
    
def auto_eval(args, task_set, final_predicted_labels, model):
    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path, f"{args.app_name}__{args.method}__score_threshold_{args.score_threshold}__autoeval_results.json")
    # output_json_path = os.path.join(args.output_path, f"google_maps__pav__score_threshold_{args.score_threshold}_auto_eval_results.json")
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path,"r") as f:
            already_data = f.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            item = json.loads(item)
            already_ids.append(item["task_id"])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_set:
        #Skip already done task
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id)
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = None
        task_description = None
        output_results = {}
        # Load results
        with open(os.path.join(args.trajectories_dir, task_id, 'results.json')) as f:
            result = json.load(f)

            task_description = result["task"]
            print("task_description : ", task_description)
            # if "episode_length" in result:
            #     episode_length = result["episode_length"]
            if "action_history" in result:
                action_history = result["action_history"]
                print("action_history : ", action_history)
            if "thoughts" in result:
                thoughts = result["thoughts"]
            if "final_result_response" in result:
                final_result_response = result["final_result_response"]
                print("final_result_response : ", final_result_response)
            if "input_image_paths" in result:
                input_image_paths = result["input_image_paths"]

        print(f"Start evaluation for {task_description}")
        # Do the auto-eval
        if args.mode == "Autonomous_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])
        
        elif args.mode == "AgentTrek_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])
        
        elif args.mode == "WebVoyager_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)
        
        elif args.mode == "WebJudge_Online_Mind2Web_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        elif args.mode == "WebJudge_general_eval":
            files = [f for f in os.listdir(trajectory_images_path) if f.lower().endswith('.png')]
            for image in sorted(files, key=extract_step):   #(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        response = model.generate(messages)[0]
        predicted_label = extract_predication(response, args.mode)
        print("predicted_label : ", predicted_label)
        
        #Store evaluation details
        evaluation_results = {"response": response, "predicted_label": predicted_label}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label

        final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("="*20)
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, f"{args.app_name}__{args.method}__score_threshold_{args.score_threshold}__autoeval_results.json"), "a+") as f_out:
        # with open(os.path.join(args.output_path, f"google_maps__pav__score_threshold_{args.score_threshold}_auto_eval_results.json"), "a+") as f_out:
            f_out.write(json.dumps(output_results) + "\n")
    
    return final_predicted_labels

def parallel_eval(args):
    #Evaluate in parallel based on num of works
    print(f"Evaluating 1 task in total.")

    #Load model
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key
    )

    args.output_path = args.image_path + '/'+ args.app_name + '/'+ args.method + '_' + args.task_number
    args.trajectories_dir = args.image_path + '/'+ args.app_name

    task_folder_name = args.method + '_' + args.task_number
    task_set = [f'{task_folder_name}']
    final_predicted_labels = []
    final_predicted_labels = auto_eval(args, task_set, final_predicted_labels, model)

    # print("final_predicted_labels : ", final_predicted_labels)
    success_num = sum(label or 0 for label in final_predicted_labels)

    print(f"Evaluation complete.\nEvaluation result : {success_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument('--mode', type=str, default='WebJudge_general_eval', help='the mode of evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--trajectories_dir", type=str, default='./qwen_3b_screenshots/google_maps', help="Path to trajectories directory")
    parser.add_argument("--api_key", type=str, default="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", help="The api key")
    parser.add_argument("--output_path", type=str, default='./WebJudge_output', help="The output path")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=1)
    args = parser.parse_args()

    parallel_eval(args)


# python -m webjudge.run_single
