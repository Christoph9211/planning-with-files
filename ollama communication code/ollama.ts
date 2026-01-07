export interface StoryContext {
  genre: string
  setting: string
  character: string
  goal: string
  model: string
  currentScene: string
  storyHistory: string[]
}

export interface StoryResponse {
  content: string
  thinking?: string
}

/**
 * Generates the opening scene of an interactive story based on given context.
 *
 * @param {Partial<StoryContext> & { retryFeedback?: string }} context
 *   A partial StoryContext object with genre, setting, character, goal, and model.
 *   If any properties are missing, they will be filled in with default values.
 *   Optionally accepts retryFeedback to nudge the regenerated opening scene.
 *
 * @returns {Promise<StoryResponse>}
 *   A StoryResponse object with the generated story content and thinking.
 */
export async function generateStoryBeginning(
  context: Partial<StoryContext> & { retryFeedback?: string },
): Promise<StoryResponse> {
  const retryFeedback = context.retryFeedback
    ? `\nUser feedback for retry: ${context.retryFeedback}\nHonor this feedback while keeping the core setup intact.`
    : ""
  const prompt = `Create the opening scene for an interactive story. First, think through your approach, then write the story.

Format your response as:
THINKING: [Your thought process about the story setup, character motivation, and scene design]
STORY: [The actual story content]

Story elements:
Genre: ${context.genre || "fantasy"}
Setting: ${context.setting || "mysterious forest"}
Main Character: ${context.character || "brave adventurer"}
Story Goal: ${context.goal || "The hero pursues their objective while the story stays focused."}
${retryFeedback}

Write an engaging opening paragraph (2-3 sentences) that sets the scene and presents the character with their first decision. End with a question or dilemma that requires a choice.`

  const response = await fetch("/api/ollama", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model: context.model || "llama3.2" }),
  })

  if (!response.ok) {
    let message = "Failed to generate story beginning"
    try {
      const errorData = await response.json()
      message = errorData.error || message
    } catch {
      message = `${message}: ${response.status} ${response.statusText}`
    }
    throw new Error(message)
  }

  const data = await response.json()
  return parseStoryResponse(data.response)
}

/**
 * Generates 3 distinct choices for what the character could do next based on the current story scene.
 *
 * @param {string} currentScene
 *   The current scene of the story.
 * @param {string} model
 *   The model to use for generating the text.
 * @param {string[]} [storyHistory]
 *   Optional array of previous scenes and actions to maintain continuity.
 *
 * @returns {Promise<{ choices: string[]; thinking?: string }>}
 *   A promise that resolves to an object with the choices and thinking.
 *   If the response is not valid, the choices will default to:
 *     ["Continue forward cautiously", "Look for another path", "Stop and assess the situation"]
 */
export async function generateStoryChoices(
  currentScene: string,
  model: string,
  storyHistory: string[] = [],
): Promise<{ choices: string[]; thinking?: string }> {
  const historyText = storyHistory.length
    ? `\nStory so far: ${storyHistory.join(" ")}`
    : ""
  const prompt = `Based on this story scene: "${currentScene}"${historyText}

First think about what would make compelling choices, then generate them.

Format your response as:
THINKING: [Your analysis of the scene and reasoning for these specific choices]
CHOICES: [The numbered list of choices]

Generate exactly 3 distinct choices for what the character could do next. Each choice should:
- Be 1-2 sentences long
- Lead to different story paths
- Be engaging and meaningful to the plot

Format the choices as:
1. [First choice]
2. [Second choice] 
3. [Third choice]`

  const response = await fetch("/api/ollama", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model }),
  })

  if (!response.ok) {
    let message = "Failed to generate choices"
    try {
      const errorData = await response.json()
      message = errorData.error || message
    } catch {
      message = `${message}: ${response.status} ${response.statusText}`
    }
    throw new Error(message)
  }

  const data = await response.json()
  const parsed = parseChoicesResponse(data.response)

  return {
    choices:
      parsed.choices.length === 3
        ? parsed.choices
        : ["Continue forward cautiously", "Look for another path", "Stop and assess the situation"],
    thinking: parsed.thinking,
  }
}

/**
 * Continue the story with the given chosen action.
 *
 * @param {StoryContext} context The context of the story so far
 * @param {string} chosenAction The action chosen by the user
 * @returns {Promise<StoryResponse>} The next scene in the story
 */
export async function continueStory(context: StoryContext, chosenAction: string): Promise<StoryResponse> {
  const history = context.storyHistory.join(" ")

  const prompt = `Continue this interactive story. First think about the consequences and story direction, then write the scene.

Format your response as:
THINKING: [Your reasoning about the consequences of this choice and how it advances the story]
STORY: [The actual story content]

Story so far: ${history}
Current scene: ${context.currentScene}
Character chose: ${chosenAction}
Story Goal: ${context.goal}

Write the next scene (2-3 sentences) showing the consequences of this choice and advancing the story. Make it engaging and set up the next decision point.`

  const response = await fetch("/api/ollama", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model: context.model }),
  })

  if (!response.ok) {
    let message = "Failed to continue story"
    try {
      const errorData = await response.json()
      message = errorData.error || message
    } catch {
      message = `${message}: ${response.status} ${response.statusText}`
    }
    throw new Error(message)
  }

  const data = await response.json()
  return parseStoryResponse(data.response)
}

/**
 * Fetches a list of available AI models from the server.
 *
 * @returns {Promise<{ models: { name: string; size: number; modified_at: string }[]; error?: string }>} 
 *   A promise that resolves to an object with the list of models and an error message if any.
 *   The models array will be empty if there was an error.
 *   The error message is either the one returned by the server or a generic message if the server is unreachable.
 */
export async function fetchAvailableModels(): Promise<{
  models: { name: string; size: number; modified_at: string }[]
  error?: string
}> {
  try {
    const response = await fetch("/api/ollama/models")

    if (!response.ok) {
      const errorData = await response.json()
      return {
        models: [],
        error: errorData.error || `Server error: ${response.status}`,
      }
    }

    const data = await response.json()
    return {
      models: data.models || [],
      error: data.error,
    }
  } catch (error) {
    console.error("Failed to fetch models:", error)
    return {
      models: [],
      error: "Cannot connect to the server. Make sure the application is running properly.",
    }
  }
}

// Helper functions to parse AI responses
function parseStoryResponse(response: string): StoryResponse {
  const thinkingMatch = response.match(/THINKING:\s*(.*?)(?=CHOICES:|$)/)
  const storyMatch = response.match(/STORY:\s*([\s\S]*)/)

  return {
    content: storyMatch ? storyMatch[1].trim() : response,
    thinking: thinkingMatch ? thinkingMatch[1].trim() : undefined,
  }
}

function parseChoicesResponse(response: string): { choices: string[]; thinking?: string } {
  const thinkingMatch = response.match(/THINKING:\s*(.*?)(?=CHOICES:|$)/)
  const choicesMatch = response.match(/CHOICES:\s*([\s\S]*)/)

  const choicesText = choicesMatch ? choicesMatch[1] : response
  const choices = choicesText
    .split("\n")
    .filter((line: string) => line.match(/^\d+\./))
    .map((line: string) => line.replace(/^\d+\.\s*/, "").trim())
    .slice(0, 3)

  return {
    choices,
    thinking: thinkingMatch ? thinkingMatch[1].trim() : undefined,
  }
}
