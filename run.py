import subprocess
import sys
import os
from colorama import Fore, Style, init

def install_requirements():
    """Install required packages from requirements.txt"""
    print(f"{Fore.YELLOW}Checking and installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(f"{Fore.GREEN}All dependencies installed successfully.")
    except Exception as e:
        print(f"{Fore.RED}Error installing dependencies: {e}")
        sys.exit(1)

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_papers(page_papers, start_idx, current_page, total_pages, total_papers):
    """Display papers with formatting"""
    clear_screen()
    
    # Display page header
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Showing papers {start_idx+1}-{start_idx+len(page_papers)} of {total_papers}")
    print(f"{Fore.GREEN}Page {current_page+1} of {total_pages}\n")
    
    # Display papers
    for i, paper in enumerate(page_papers):
        print(f"{Fore.CYAN}{start_idx+i+1}. {Style.BRIGHT}{paper['Title']}")
        print(f"{Fore.MAGENTA}   DOI: {paper['DOI']}")
        print(f"{Fore.GREEN}   Citations: {paper['Citation Count']} | Year: {paper['Year']}")
        print(f"{Fore.BLUE}   Relevance Score: {round(paper['Final Score'] * 100, 2)}%")
        
        # Format abstract with word wrap for better readability
        abstract = paper['Abstract'] or 'N/A'
        print(f"{Fore.WHITE}   Abstract: {abstract}\n")

def display_navigation_help(current_page, total_pages):
    """Display navigation instructions"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Navigation Options:")
    
    # Only show relevant navigation options based on current page
    if current_page < total_pages - 1:
        print(f"{Fore.YELLOW}n - Next page")
    
    if current_page > 0:
        print(f"{Fore.YELLOW}p - Previous page")
    
    print(f"{Fore.YELLOW}s - New search")
    print(f"{Fore.YELLOW}q - Exit program")

def get_user_choice():
    """Get user choice for navigation"""
    choice = input(f"\n{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").strip().lower()
    return choice

def main():
    # Initialize colorama
    init(autoreset=True)
    
    # Install dependencies
    try:
        install_requirements()
    except Exception as e:
        print(f"{Fore.RED}Error installing dependencies: {e}")
        return
    
    # Import the paper search functionality
    from main import get_ranked_papers, PaperDisplay
    
    # Print welcome message
    clear_screen()
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'DeepCite Research Paper Search Tool':^60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}\n")
    
    running = True
    while running:
        try:
            # Get user query
            query = input(f"\n{Fore.YELLOW}Enter your research query: {Style.RESET_ALL}")
            
            if not query.strip():
                print(f"{Fore.RED}Query cannot be empty. Please try again.")
                continue
                
            print(f"\n{Fore.CYAN}Searching for papers on: '{query}'...\n")
            ranked_papers = get_ranked_papers(query, limit=100)
            
            if isinstance(ranked_papers, str):
                print(f"{Fore.RED}{ranked_papers}")
                
                # Wait for user input
                choice = input(f"\n{Fore.YELLOW}Press 's' to try another search or 'q' to exit: {Style.RESET_ALL}").strip().lower()
                if choice == 'q':
                    running = False
                continue
            
            # Create paper display for pagination
            paper_display = PaperDisplay(ranked_papers)
            
            # Navigation loop
            navigating = True
            while navigating and running:
                # Get papers for current page
                page_papers, start_idx, end_idx = paper_display.get_page_papers()
                
                # Display the current page of papers
                display_papers(
                    page_papers, 
                    start_idx,
                    paper_display.current_page, 
                    paper_display.total_pages,
                    len(paper_display.papers)
                )
                
                # Show navigation options
                display_navigation_help(paper_display.current_page, paper_display.total_pages)
                
                # Get user choice
                choice = get_user_choice()
                
                if choice == 'n':
                    # Only process next page if we're not on the last page
                    if paper_display.current_page < paper_display.total_pages - 1:
                        paper_display.next_page()
                elif choice == 'p':
                    # Only process previous page if we're not on the first page
                    if paper_display.current_page > 0:
                        paper_display.prev_page()
                elif choice == 's':
                    navigating = False  # Break navigation loop to start new search
                    clear_screen()
                elif choice == 'q':
                    navigating = False
                    running = False
                else:
                    print(f"{Fore.RED}Invalid choice. Press any key to continue...")
                    input()
        
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Interrupted by user.")
            running = False
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {e}")
            
            # Wait for user input
            choice = input(f"\n{Fore.YELLOW}Press 's' to try again or 'q' to exit: {Style.RESET_ALL}").strip().lower()
            if choice == 'q':
                running = False
    
    print(f"\n{Fore.GREEN}Thank you for using DeepCite Research Paper Search Tool!")

if __name__ == "__main__":
    main()