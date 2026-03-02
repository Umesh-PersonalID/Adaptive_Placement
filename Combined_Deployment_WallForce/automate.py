#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import re
from datetime import datetime

def run_map_deployment(map_num):
    """Run deployment for a specific map and extract results"""
    map_dir = f"Maps/map{map_num}"
    script_name = f"greedy+potential_map{map_num}.py"
    script_path = os.path.join(map_dir, script_name)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING MAP {map_num}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return None
    
    # Change to map directory
    original_dir = os.getcwd()
    os.chdir(map_dir)
    
    try:
        # Run the deployment script
        print(f"Starting deployment for map{map_num}...")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True)  # No timeout
        
        end_time = time.time()
        execution_time = end_time - start_time  
        
        if result.returncode == 0:
            print(f"✅ Map{map_num} completed successfully in {execution_time:.2f}s")
            
            # Extract results from output
            output = result.stdout
            results = extract_results(output, map_num, execution_time)
            
            # Save first frame if animation was generated
            save_first_frame(map_num)
            
            return results
        else:
            print(f"❌ Map{map_num} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return {
                'map': map_num,
                'status': 'FAILED',
                'error': result.stderr,
                'execution_time': execution_time
            }
            
    except Exception as e:
        print(f" Map{map_num} crashed: {str(e)}")
        return {
            'map': map_num,
            'status': 'CRASHED',
            'error': str(e),
            'execution_time': time.time() - start_time
        }
    finally:
        # Return to original directory
        os.chdir(original_dir)

def extract_results(output, map_num, execution_time):
    """Extract key metrics from the script output"""
    results = {
        'map': map_num,
        'status': 'SUCCESS',
        'execution_time': execution_time
    }
    
    try:
        # Extract final metrics using regex
        coverage_match = re.search(r'Coverage:\s*([\d.]+)%', output)
        overlap_match = re.search(r'Overlap:\s*([\d.]+)%', output)
        cost_match = re.search(r'Final Cost:\s*([\d.]+)', output)
        nodes_match = re.search(r'Deployed Nodes:\s*(\d+)', output)
        
        # Extract stopping condition info
        if "Stopping condition reached" in output:
            results['stopped_by'] = 'COST_THRESHOLD'
            threshold_match = re.search(r'Cost ([\d.]+) exceeds threshold', output)
            if threshold_match:
                results['final_cost'] = float(threshold_match.group(1))
        elif "Coverage threshold reached" in output:
            results['stopped_by'] = 'COVERAGE_THRESHOLD'
        elif "No more frontiers" in output:
            results['stopped_by'] = 'NO_FRONTIERS'
        else:
            results['stopped_by'] = 'COMPLETED'
        
        # Store extracted values
        results['coverage'] = float(coverage_match.group(1)) if coverage_match else 0.0
        results['overlap'] = float(overlap_match.group(1)) if overlap_match else 0.0
        results['final_cost'] = float(cost_match.group(1)) if cost_match else 0.0
        results['deployed_nodes'] = int(nodes_match.group(1)) if nodes_match else 0
        
        # Extract cost evolution
        cost_evolution_match = re.search(r'Cost evolution: \[(.*?)\]', output)
        if cost_evolution_match:
            cost_str = cost_evolution_match.group(1)
            costs = [float(c.strip("'")) for c in cost_str.split(', ') if c.strip("'")]
            results['cost_evolution'] = costs
        
        # Count node deployment iterations
        node_iterations = output.count('Node ')
        results['iterations'] = node_iterations
        
    except Exception as e:
        print(f"Warning: Could not extract all results for map{map_num}: {e}")
        results['extraction_error'] = str(e)
    
    return results

def save_first_frame(map_num):
    """Save the first frame (initial hexagonal placement) as an image"""
    try:
        # This would need to be implemented in the main script
        # For now, we'll just note that Final_Deployment.png exists
        if os.path.exists("Final_Deployment.png"):
            print(f"📸 Final deployment image saved for map{map_num}")
        
        # Check if position files exist
        if os.path.exists("Final_Position.txt"):
            print(f"📍 Final positions saved for map{map_num}")
            
    except Exception as e:
        print(f"Warning: Could not save first frame for map{map_num}: {e}")

def save_consolidated_results(all_results):
    """Save all results to a consolidated text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"consolidated_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("COST-BASED DEPLOYMENT EXPERIMENT RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Maps Processed: 14-22 ({len(all_results)} maps)\n")
        f.write(f"Cost Threshold: 5.0\n")
        f.write(f"Hexagonal Density: ~50% (every other coordinate)\n")
        f.write("="*60 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Map':<6} {'Status':<12} {'Nodes':<6} {'Coverage':<10} {'Overlap':<9} {'Cost':<8} {'Time(s)':<8} {'Stopped By':<15}\n")
        f.write("-" * 80 + "\n")
        
        total_time = 0
        successful_maps = 0
        
        for result in all_results:
            if result:
                status = result.get('status', 'UNKNOWN')
                nodes = result.get('deployed_nodes', 0)
                coverage = result.get('coverage', 0.0)
                overlap = result.get('overlap', 0.0)
                cost = result.get('final_cost', 0.0)
                exec_time = result.get('execution_time', 0.0)
                stopped_by = result.get('stopped_by', 'UNKNOWN')
                
                f.write(f"{result['map']:<6} {status:<12} {nodes:<6} {coverage:<10.2f} {overlap:<9.2f} {cost:<8.4f} {exec_time:<8.1f} {stopped_by:<15}\n")
                
                total_time += exec_time
                if status == 'SUCCESS':
                    successful_maps += 1
        
        f.write("-" * 80 + "\n")
        f.write(f"Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"Successful Maps: {successful_maps}/{len(all_results)}\n")
        
        # Detailed results
        f.write(f"\n\nDETAILED RESULTS\n")
        f.write("="*60 + "\n")
        
        for result in all_results:
            if result:
                f.write(f"\nMAP {result['map']}:\n")
                f.write(f"  Status: {result.get('status', 'UNKNOWN')}\n")
                f.write(f"  Execution Time: {result.get('execution_time', 0):.2f}s\n")
                f.write(f"  Deployed Nodes: {result.get('deployed_nodes', 0)}\n")
                f.write(f"  Final Coverage: {result.get('coverage', 0):.2f}%\n")
                f.write(f"  Final Overlap: {result.get('overlap', 0):.2f}%\n")
                f.write(f"  Final Cost: {result.get('final_cost', 0):.4f}\n")
                f.write(f"  Stopping Condition: {result.get('stopped_by', 'UNKNOWN')}\n")
                f.write(f"  Iterations: {result.get('iterations', 0)}\n")
                
                if 'cost_evolution' in result:
                    f.write(f"  Cost Evolution: {result['cost_evolution']}\n")
                
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                
                if 'extraction_error' in result:
                    f.write(f"  Extraction Warning: {result['extraction_error']}\n")
    
    print(f"\n📊 Results saved to: {results_file}")
    return results_file

def main():
    """Main automation function"""
    print("🚀 STARTING COST-BASED DEPLOYMENT AUTOMATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Verify we're in the right directory
    if not os.path.exists("Maps"):
        print("❌ ERROR: Maps directory not found!")
        print("Please run this script from the Combined_Deployment_Copy directory")
        sys.exit(1)
    
    # Check which maps exist
    available_maps = []
    map_nums = [14,15,16,17,18,19,20,21,22]
    for map_num in map_nums:
        script_path = f"Maps/map{map_num}/greedy+potential_map{map_num}.py"
        if os.path.exists(script_path):
            available_maps.append(map_num)
        else:
            print(f"⚠️  Warning: Map{map_num} script not found")
    
    print(f"📍 Found {len(available_maps)} maps: {available_maps}")
    
    if not available_maps:
        print("❌ ERROR: No map scripts found!")
        sys.exit(1)
    
    # Run deployments for all available maps
    all_results = []
    overall_start = time.time()
    
    for map_num in available_maps:
        try:
            result = run_map_deployment(map_num)
            all_results.append(result)
            
            # Brief pause between maps
            time.sleep(2)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user during map{map_num}")
            break
        except Exception as e:
            print(f"💥 Unexpected error processing map{map_num}: {e}")
            all_results.append({
                'map': map_num,
                'status': 'ERROR',
                'error': str(e),
                'execution_time': 0
            })
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Save consolidated results
    results_file = save_consolidated_results(all_results)
    
    # Final summary
    print(f"\n🎯 AUTOMATION COMPLETED")
    print(f"📊 Processed: {len(all_results)} maps")
    print(f"⏱️  Total Time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"📄 Results File: {results_file}")
    
    # Count successful runs
    successful = sum(1 for r in all_results if r and r.get('status') == 'SUCCESS')
    print(f"✅ Successful: {successful}/{len(all_results)} maps")
    
    if successful < len(all_results):
        print(f"⚠️  Check the results file for detailed error information")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n🛑 Automation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
